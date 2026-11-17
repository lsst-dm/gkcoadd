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

__all__ = ("WarpedImage",)

from collections.abc import Sequence
from types import EllipsisType
from typing import Any, cast

import astropy.units
import numpy as np

from lsst.images import Box, Image, Mask, MaskedImage, MaskSchema, SkyProjection, TractFrame
from lsst.images.cells import CellGrid, CellGridBounds, CellPointSpreadFunction, PatchDefinition


# TODO[deferred]: someday, this class should move to lsst.images.  But that's
# only after the algorithm that produces it is completed and stabilized for
# inclusion in the official pipelines.
class WarpedImage(MaskedImage):
    """A single-visit, single-detector image warped onto a coadd patch.

    Parameters
    ----------
    image
        The main science image.
    mask
        A bitmask image that annotates the main image plane.
    variance
        The per-pixel variance of the main image.  Note that the full
        uncertainty includes covariance, which is not tracked per pixel.
    coverage
        The kernel-weighted fraction of the input support that fell on
        good pixels (i.e. a warp of a synthetic input image that is
        ``1`` for good pixels and ``0`` for bad pixels).  Must be
        dimensionless with the same bounding box as ``image`` if provided.
        Values default to ``0.0``, meaning no data.
    image_guess
        The kernel-weighted average of the input image over bad pixels only
        (i.e. a warp of a synthetic input image that is ``0`` for good pixels
        and the current best guess for bad pixels).  Must have the same units
        and bounding box as ``image`` if provided.  Values default to ``0.0``.
    noise_realizations
        A sequence of images with Monte Carlo realizations of the noise in
        the image.  Each must have the same units and bounding box as
        ``image``.
    noise_guesses
        A sequence of images analogous to ``image_guess``, with elements
        corresponding to ``noise_realizations``.  May be empty if
        ``noise_realizations`` is not, in which case each element defaults to
        an image of zeros.
    mask_schema
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    sky_projection
        Projection that maps the pixel grid to the sky.  Can only be `None` if
        a projection is already attached to ``image``.
    band
        Name of the band.
    psf
        Point-spread function model, processed identically to ``image``.  The
        missing cells reported by ``psf.bounds`` are assumed to apply to all
        image data for that cell as well (i.e. there is a PSF for a cell if
        and only if there is image data for that cell).
    warp_kernel
        The transpose of the actual kernel used to warp+convolve ``image``,
        rendered on the detector pixel grid and then processed the same way as
        the other image planes.  This is used to whiten the noise on the
        coadd.
    patch
        Identifiers and geometry of the full patch, if the image is confined
        to a single patch.  When present, the cell grid of the PSF and
        provenance (if provideD) must be the full patch grid, even if its
        bounds select a subset of that area.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        coverage: Image | None = None,
        image_guess: Image | None = None,
        noise_realizations: Sequence[Image] = (),
        noise_guesses: Sequence[Image] = (),
        mask_schema: MaskSchema | None = None,
        sky_projection: SkyProjection[Any] | None = None,
        band: str | None = None,
        psf: CellPointSpreadFunction,
        warp_kernel: CellPointSpreadFunction,
        patch: PatchDefinition | None = None,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            sky_projection=sky_projection,
        )
        if self.image.unit is None:
            raise TypeError("The image component of a WarpedImage must have units.")
        if self.image.sky_projection is None:
            raise TypeError("The sky_projection component of a WarpedImage cannot be None.")
        if not isinstance(self.image.sky_projection.pixel_frame, TractFrame):
            raise TypeError("The sky_projection's pixel frame must be a TractFrame for WarpedImage.")
        if len(noise_guesses) not in (0, len(noise_realizations)):
            raise ValueError(
                "The noise_guesses sequence must be empty or have the same length as noise_realizations."
            )
        self._coverage = self._init_plane(coverage, name="coverage", unit=None)
        self._image_guess = self._init_plane(image_guess, name="image_guess", unit=self.image.unit)
        self._noise_realizations = [
            self._init_plane(v, name="noise_realizations", unit=self.image.unit) for v in noise_realizations
        ]
        self._noise_guesses = [
            self._init_plane(v, name="noise_guesses", unit=self.image.unit)
            for v in (noise_guesses or (None,) * len(noise_realizations))
        ]
        self._band = band
        if psf.bounds.bbox != self.bbox:
            psf = psf[self.bbox.intersection(psf.bounds.bbox)]
        self._psf = psf
        self._warp_kernel = warp_kernel
        self._patch = patch

    def _init_plane(self, plane: Image | None, *, name: str, unit: astropy.units.UnitBase | None) -> Image:
        """Check an additional image plane against the main image, or replace a
        missing one with a default image of zeros.

        Parameters
        ----------
        plane
            The image plane to check, or `None` for the default.  Zeros are
            appropriate for missing data: no coverage and no guess.
        name
            Name of the plane, used in error messages.
        unit
            Units the plane must have, matching the main image.  `None` means
            the plane must be dimensionless.

        Returns
        -------
        plane : `~lsst.images.Image`
            The plane, with its sky projection replaced by this image's.
        """
        if plane is None:
            return Image(
                0.0,
                dtype=np.float32,
                bbox=self.bbox,
                unit=unit,
                sky_projection=self.sky_projection,
            )
        if plane.bbox != self.bbox:
            raise ValueError(f"Image ({self.bbox}) and {name} ({plane.bbox}) bboxes do not agree.")
        if unit is None:
            if plane.unit is not None and plane.unit != astropy.units.dimensionless_unscaled:
                raise ValueError(f"{name} must be dimensionless, not {plane.unit}.")
        elif plane.unit is None:
            plane = plane.view(unit=unit)
        elif plane.unit != unit:
            raise ValueError(f"{name} unit ({plane.unit}) should be the same as the image's ({unit}).")
        return plane.view(sky_projection=self.sky_projection)

    @property
    def skymap(self) -> str:
        """Name of the skymap (`str`)."""
        return self.sky_projection.pixel_frame.skymap

    @property
    def tract(self) -> int:
        """ID of the tract (`int`)."""
        return self.sky_projection.pixel_frame.tract

    @property
    def patch(self) -> PatchDefinition:
        """Identifiers and geometry of the full patch, if the image is confined
        to a single patch (`PatchDefinition`).
        """
        if self._patch is None:
            raise AttributeError("Coadd has no patch information.")
        return self._patch

    @property
    def band(self) -> str | None:
        """Name of the band (`str` or `None`)."""
        return self._band

    @property
    def noise_realizations(self) -> Sequence[Image]:
        """A sequence of images with Monte Carlo realizations of the noise in
        the coadd (`~collections.abc.Sequence` [`.Image`]).
        """
        return self._noise_realizations

    @property
    def unit(self) -> astropy.units.UnitBase:
        """The units of the image plane (`astropy.units.Unit`)."""
        return cast(astropy.units.UnitBase, super().unit)

    @property
    def sky_projection(self) -> SkyProjection[TractFrame]:
        """The projection that maps the pixel grid to the sky
        (`.SkyProjection` [`.TractFrame`]).
        """
        return cast(SkyProjection[TractFrame], super().sky_projection)

    @property
    def psf(self) -> CellPointSpreadFunction:
        """Point-spread function model, processed identically to `image`
        (`CellPointSpreadFunction`).
        """
        return self._psf

    @property
    def warp_kernel(self) -> CellPointSpreadFunction:
        """The transpose of the actual kernel used to warp+convolve ``image``,
        rendered on the detector pixel grid and then processed the same way as
        the other image planes (`CellPointSpreadFunction`).

        This is used to whiten the noise on the
        """
        return self._warp_kernel

    @property
    def bounds(self) -> CellGridBounds:
        """The grid of cells that overlap this coadd and a set of missing
        cells (`CellGridBounds`).
        """
        return self._psf.bounds

    @property
    def grid(self) -> CellGrid:
        """The grid of cells that overlap this coadd (`CellGrid`)."""
        return self._psf.bounds.grid

    def __getitem__(self, bbox: Box | EllipsisType) -> WarpedImage:
        bbox, _ = self._handle_getitem_args(bbox)
        return self._transfer_metadata(
            WarpedImage(
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
                sky_projection=self.sky_projection,
                coverage=self._coverage[bbox],
                image_guess=self._image_guess[bbox],
                noise_realizations=[v[bbox] for v in self._noise_realizations],
                noise_guesses=[v[bbox] for v in self._noise_guesses],
                band=self.band,
                psf=self.psf[bbox],
                warp_kernel=self.warp_kernel[bbox],
                patch=self._patch,
            ),
            bbox=bbox,
        )

    # TODO[deferred]: add a _describe implementation

    def copy(self) -> WarpedImage:
        """Deep-copy the warp."""
        return self._transfer_metadata(
            WarpedImage(
                image=self._image.copy(),
                mask=self._mask.copy(),
                variance=self._variance.copy(),
                sky_projection=self.sky_projection,
                coverage=self._coverage.copy(),
                image_guess=self._image_guess.copy(),
                noise_realizations=[v.copy() for v in self._noise_realizations],
                noise_guesses=[v.copy() for v in self._noise_guesses],
                band=self.band,
                psf=self.psf,
                warp_kernel=self.warp_kernel,
                patch=self._patch,
            ),
            copy=True,
        )

    # TODO[deferred]: implement serialization hooks
