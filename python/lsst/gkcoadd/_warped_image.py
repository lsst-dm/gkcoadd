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

__all__ = ("BaseWarpImage", "MissingCellReason", "WarpFields", "WarpedImage")

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Self, TypedDict, cast, final

import astropy.units
from astro_metadata_translator import ObservationInfo

from lsst.images import Box, Image, Mask, MaskedImage, SkyProjection, TractFrame
from lsst.images.cells import CellGrid, CellGridBounds, PatchDefinition

from ._cell_kernel import CellKernel, CellKernelWorkspace


class WarpFields[K: CellKernel | CellKernelWorkspace | Image](TypedDict):
    """A dictionary for the minimal fields of a warp or to-be-warped visit
    image.
    """

    image: Image
    mask: Mask
    variance: Image
    coverage: Image
    image_guess: Image
    noise_realizations: list[Image]
    noise_guesses: list[Image]
    psf: K
    warp_kernel: K


class MissingCellReason(enum.StrEnum):
    """Enumeration of the reasons cells may be missing from a warp."""

    NO_COARSE_OVERLAP = enum.auto()
    """This cell did not overlap the {visit, detector} image in even coarse
    conservative polygonal overlap checks.
    """

    UNEXPECTED_OVERLAP_GEOMETRY = enum.auto()
    """The geometry of the overlap between the {visit, detector} was not
    expected (e.g. not a convex polygon).
    """

    FAILED_PSF_EVALUATION = enum.auto()
    """The PSF model could not be evaluated for this cell."""

    FAILED_PSF_MOMENTS = enum.auto()
    """The PSF model's moments could not be computed or had suspicious values.
    """

    PSF_TOO_LARGE = enum.auto()
    """The PSF was too large."""


class BaseWarpImage[K: CellKernel | CellKernelWorkspace | Image](MaskedImage, ABC):
    """A single-visit, single-detector image with the planes needed for
    warping and coaddition.

    Parameters
    ----------
    fields
        The fields for this object.  Fields that are part of MaskedImage or
        always images are checked for consistency; ``psf`` and ``warp_kernel``
        correctness is the responsibility of the (subclass) caller.

    Notes
    -----
    This is an intermediate base class intended for code reuse, not a public
    interface or a concrete type.
    """

    def __init__(self, fields: WarpFields):
        super().__init__(
            image=fields["image"],
            mask=fields["mask"],
            variance=fields["variance"],
        )
        if self.image.unit != astropy.units.nJy:
            raise TypeError(
                "The image component of an image that will be or has been warped must have nJy units."
            )
        if self.image.sky_projection is None:
            raise TypeError(
                "The sky_projection component of an image that will be or has been warped cannot be None."
            )
        for name, image_field in self._iter_extra_image_fields(fields):
            # TODO[agent]: make sure exception types align similar lsst.images
            # existing checks.
            if image_field.bbox != self.image.bbox:
                raise ValueError(
                    f"{name}.bbox {image_field.bbox} does not match the image plane's {self.image.bbox}."
                )
            if image_field.sky_projection != self.image.sky_projection:
                raise ValueError(f"{name}.sky_projection does not match the image plane's sky projection.")
            if name == "coverage":
                if image_field.unit is not None:
                    raise ValueError("coverage plane must have unit=None.")
            else:
                if image_field.unit != self.image.unit:
                    raise ValueError(
                        f"{name}.unit {image_field.unit} does not match the image plane's {self.image.unit}."
                    )
        self._coverage = fields["coverage"]
        self._image_guess = fields["image_guess"]
        self._noise_realizations = list(fields["noise_realizations"])
        self._noise_guesses = list(fields["noise_guesses"])
        if len(self._noise_realizations) != len(self._noise_guesses):
            raise ValueError(
                f"Number of noise_realizations ({len(self._noise_realizations)}) and "
                f"noise_guesses ({len(self._noise_guesses)}) do not agree."
            )
        self._psf = fields["psf"]
        self._warp_kernel = fields["warp_kernel"]

    @property
    def coverage(self) -> Image:
        """Whether this pixel has valid data (1.0) or not (0.0), or the
        kernel-weighted average of such an image post-warp
        (`lsst.images.Image`).
        """
        return self._coverage

    @property
    def image_guess(self) -> Image:
        """Zero for valid data pixels, and the same value as `image` otherwise,
        or the kernel-weighted average of such an image post-warp
        (`lsst.images.Image`).
        """
        return self._image_guess

    @property
    def noise_realizations(self) -> Sequence[Image]:
        """A sequence of images with Monte Carlo realizations of the noise in
        the coadd (`~collections.abc.Sequence` [`lsst.images.Image`]).
        """
        return self._noise_realizations

    @property
    def noise_guesses(self) -> Sequence[Image]:
        """A sequence of images analogous to `image_guess`, with elements
        corresponding to `noise_realizations`
        (`~collections.abc.Sequence` [`lsst.images.Image`]).
        """
        return self._noise_guesses

    @property
    def psf(self) -> K:
        """The point-spread function of the image."""
        return self._psf

    @property
    def warp_kernel(self) -> K:
        """The approximation to the point-spread function used as the warping
        kernel.
        """
        return self._warp_kernel

    @property
    def unit(self) -> astropy.units.UnitBase:
        """The units of the image plane (`astropy.units.Unit`)."""
        return cast(astropy.units.UnitBase, super().unit)

    def copy(self) -> Self:
        return self._transfer_metadata(
            self._make_modified(
                {
                    "image": self._image.copy(),
                    "mask": self._mask.copy(),
                    "variance": self._variance.copy(),
                    "coverage": self._coverage.copy(),
                    "image_guess": self._image_guess.copy(),
                    "noise_realizations": [v.copy() for v in self._noise_realizations],
                    "noise_guesses": [v.copy() for v in self._noise_guesses],
                    "psf": self._psf.copy(),
                    "warp_kernel": self._warp_kernel.copy(),
                }
            ),
            copy=True,
        )

    def __getitem__(self, bbox: Box | EllipsisType) -> Self:
        bbox, _ = self._handle_getitem_args(bbox)
        fields = self._slice_image_fields(bbox)
        fields["psf"] = self._slice_kernel(fields["psf"], bbox)
        fields["warp_kernel"] = self._slice_kernel(fields["warp_kernel"], bbox)
        return self._transfer_metadata(self._make_modified(fields), bbox=bbox)

    @staticmethod
    def _iter_extra_image_fields(fields: WarpFields[K]) -> Iterator[tuple[str, Image]]:
        for name in ("coverage", "image_guess"):
            yield name, fields[name]
        for name in ("noise_realizations", "noise_guesses"):
            for n, image in enumerate(fields[name]):  # type: ignore[literal-required]
                yield f"{name}[{n}]", image

    @abstractmethod
    def _make_modified(self, fields: WarpFields) -> Self:
        raise NotImplementedError()

    @final
    def _slice_image_fields(self, bbox: Box) -> WarpFields:
        """Apply bounding-box slicing to the image-like fields.

        The ``psf`` and ``warp_kernel`` fields are returned unmodified.
        """
        return {
            "image": self.image[bbox],
            "mask": self.mask[bbox],
            "variance": self.variance[bbox],
            "coverage": self._coverage[bbox],
            "image_guess": self._image_guess[bbox],
            "noise_realizations": [v[bbox] for v in self._noise_realizations],
            "noise_guesses": [v[bbox] for v in self._noise_guesses],
            "psf": self._psf,
            "warp_kernel": self._warp_kernel,
        }

    def _slice_kernel(self, kernel: K, bbox: Box) -> K:
        raise TypeError("Box slicing is not supported by this warp variant.")


@final
class WarpedImage(BaseWarpImage[CellKernel]):
    """A single-visit, single-detector image warped onto a coadd patch.

    Parameters
    ----------
    fields
        The inherited fields for this object: all image and mask planes, and
        the psf and warp kernel objects.
    obs_info
        Metadata about the observation.
    band
        Name of the band.
    patch
        Identifiers and geometry of the full patch, if the image is confined
        to a single patch.  When present, the cell grid of the PSF and
        provenance (if provided) must be the full patch grid, even if its
        bounds select a subset of that area.
    """

    def __init__(
        self, fields: WarpFields[CellKernel], obs_info: ObservationInfo, band: str, patch: PatchDefinition
    ):
        super().__init__(fields)
        assert self.image.sky_projection is not None, "Guaranteed by super().__init__"
        if not isinstance(self.image.sky_projection.pixel_frame, TractFrame):
            raise TypeError("The sky_projection's pixel frame must be a TractFrame for WarpedImage.")
        self._obs_info = obs_info
        self._band = band
        self._patch = patch

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
        """Identifiers and geometry of the full patch (`PatchDefinition`)."""
        return self._patch

    @property
    def sky_projection(self) -> SkyProjection[TractFrame]:
        """The projection that maps the pixel grid to the sky
        (`lsst.images.SkyProjection` [`lsst.images.TractFrame`]).
        """
        return cast(SkyProjection[TractFrame], super().sky_projection)

    @property
    def obs_info(self) -> ObservationInfo:
        """General information about this observation in standard form.
        (`~astro_metadata_translator.ObservationInfo`).
        """
        return self._obs_info

    @property
    def physical_filter(self) -> str:
        """Full name of the physical bandpass filter (`str`)."""
        assert self._obs_info.physical_filter is not None, "Guaranteed at construction."
        return self._obs_info.physical_filter

    @property
    def band(self) -> str:
        """Short name of the bandpass filter (`str`)."""
        return self._band

    @property
    def bounds(self) -> CellGridBounds:
        """The grid of cells that overlap this image and a set of missing
        cells (`lsst.images.cells.CellGridBounds`).
        """
        return self._psf.bounds

    @property
    def grid(self) -> CellGrid:
        """The patch-level grid of cells (`lsst.images.cells.CellGrid`)."""
        return self._psf.bounds.grid

    def _make_modified(self, fields: WarpFields) -> Self:
        return WarpedImage(fields, obs_info=self.obs_info, band=self.band, patch=self.patch)

    def _slice_kernel(self, kernel: CellKernel, bbox: Box) -> CellKernel:
        return kernel[bbox]

    # TODO[deferred]: add a _describe implementation
    # TODO[deferred]: implement serialization hooks
