# This file is part of gkcoadd.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from collections.abc import Set
from typing import overload

__all__ = ("CellKernel", "CellKernelWorkspace")

from functools import cached_property

import numpy as np

from lsst.images import YX, BoundsError, Box, Image
from lsst.images.cells import CellGrid, CellGridBounds, CellIJ


class CellKernelWorkspace:
    """A mutable convolution kernel that is piecewise constant over cells on a
    full patch grid, with no knowledge of missing cells.

    Parameters
    ----------
    array
        A 4-d array of kernel images with shape
        ``grid.grid_size.as_tuple() + kernel_bbox.shape``.
    grid
        Definition of the full cell grid.
    """

    def __init__(self, array: np.ndarray, grid: CellGrid) -> None:
        self._array = array
        self._grid = grid

    @classmethod
    def make_empty(cls, kernel_shape: YX[int], grid: CellGrid) -> CellKernelWorkspace:
        """Construct an empty workspace with all kernel values set to NaN.

        Parameters
        ----------
        kernel_shape
            Shape of the kernel images.
        grid
            Definition of the full cell grid.
        """
        array = np.full(grid.grid_size.as_tuple() + kernel_shape, np.nan, dtype=np.float32)
        return cls(array, grid)

    @property
    def grid(self) -> CellGrid:
        """The grid that defines the kernel's cells (`CellGrid`)."""
        return self._grid

    @cached_property
    def kernel_bbox(self) -> Box:
        sy, sx = self._array.shape[2:]
        ry = sy // 2
        rx = sx // 2
        return Box.factory[-ry : ry + 1, -rx : rx + 1]

    def __getitem__(self, key: CellIJ) -> Image:
        match key:
            case CellIJ():
                if key not in self._grid:
                    raise BoundsError(f"Cell {key} is out of bounds for this kernel.")
                return Image(self._array[key.i, key.j], bbox=self.kernel_bbox)
            case _:
                raise TypeError(f"Invalid argument for CellKernelWorkspace.__getitem__: {key}")

    def copy(self) -> CellKernelWorkspace:
        """Return a deep copy of this kernel."""
        return CellKernelWorkspace(self._array.copy(), self._grid)

    def finish(self, missing: Set[CellIJ]) -> CellKernel:
        """Finalize this object and convert it into a `CellKernel`.

        The returned object takes ownership of the underlying array and marks
        it as read-only.
        """
        bounds = CellGridBounds(grid=self.grid, bbox=self.grid.bbox, missing=frozenset(missing))
        # TODO[DM-57096]: if CellGridBounds shrinks its bbox to non-missing
        # cells, this will need to adapt.
        return CellKernel(self._array, bounds)


class CellKernel:
    """An immutable convolution kernel that is piecewise constant over cells
    on a grid, with some cells potentially missing.

    Parameters
    ----------
    array
        A 4-d array of kernel images with shape
        ``bounds.subgrid_size.as_tuple() + kernel_bbox.shape``.  The kernel
        takes ownership of this array and flags it as read-only; external code
        that keeps a reference to it must not modify it.
    bounds
        Description of the cell grid and any missing cells.  Array entries for
        missing cells within the subgrid should be NaN.
    """

    def __init__(
        self,
        array: np.ndarray,
        bounds: CellGridBounds,
    ) -> None:
        self._array = array
        self._array.flags.writeable = False
        self._bounds: CellGridBounds = bounds

    @property
    def grid(self) -> CellGrid:
        """The grid that defines the kernel's cells (`CellGrid`)."""
        return self._bounds.grid

    @property
    def bounds(self) -> CellGridBounds:
        """The bounds where the kernel can be evaluated (`CellGridBounds`)."""
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        sy, sx = self._array.shape[2:]
        ry = sy // 2
        rx = sx // 2
        return Box.factory[-ry : ry + 1, -rx : rx + 1]

    @overload
    def __getitem__(self, bbox: Box) -> CellKernel: ...
    @overload
    def __getitem__(self, index: CellIJ) -> Image: ...

    def __getitem__(self, key: Box | CellIJ) -> CellKernel | Image:
        match key:
            case CellIJ():
                if key in self._bounds.missing:
                    raise BoundsError(f"Cell {key} is missing for this kernel.")
                if not self._bounds.contains_cell(key):
                    raise BoundsError(f"Cell {key} is out of bounds for this kernel.")
                index = key - self._bounds.subgrid_start
                try:
                    return Image(self._array[index.i, index.j], bbox=self.kernel_bbox)
                except IndexError:
                    raise BoundsError(f"Cell {key} is out of bounds for this kernel.")
            case Box():
                bounds, slices = self._subset_impl(self._bounds, key)
                return CellKernel(self._array[slices.y, slices.x, ...], bounds=bounds)
            case _:
                raise TypeError(f"Invalid argument for CellKernel.__getitem__: {key}")

    def copy(self) -> CellKernel:
        """Return a deep copy of this kernel.

        Notes
        -----
        The underlying array is read-only, which would usually make a deep
        copy unnecessary.  But slicing can leave a small kernel holding a
        reference to a much larger array, and copying is the only way to free
        up that memory if it is unused elsewhere.
        """
        return CellKernel(self._array.copy(), self._bounds)

    @staticmethod
    def _subset_impl(bounds: CellGridBounds, bbox: Box) -> tuple[CellGridBounds, YX[slice]]:
        subset_bounds = bounds[bbox]
        start = subset_bounds.subgrid_start - bounds.subgrid_start
        stop = subset_bounds.subgrid_stop - bounds.subgrid_start
        return subset_bounds, YX(y=slice(start.i, stop.i), x=slice(start.j, stop.j))
