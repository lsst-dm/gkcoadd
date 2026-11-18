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

from collections.abc import Sequence

from lsst.utils.logging import LsstLogAdapter

__all__ = ("GaussianKaiserWarpConfig", "GaussianKaiserWarpTask")

import math
from typing import Any, ClassVar

import astropy.units
import numpy as np

from lsst.images import (
    XY,
    YX,
    Bounds,
    Box,
    DetectorFrame,
    Ellipse,
    Image,
    Mask,
    MaskPlane,
    MaskSchema,
    NoOverlapError,
    Polygon,
    SkyProjection,
    TractFrame,
    Transform,
    VisitImage,
)
from lsst.images.cells import CellGridBounds, CellIJ, CellPointSpreadFunction, PatchDefinition
from lsst.pex.config import Config, DictField, Field
from lsst.pipe.base import NoWorkFound, Task

from ._warped_image import BaseWarpImage, WarpedImage


class GaussianKaiserWarpConfig(Config):
    """Configuration for GaussianKaiserWarpTask."""

    transform_atol = Field[float](
        "Absolute tolerance in detector pixels for approximating the transform "
        "from the tract frame to the detector frame.",
        default=0.005,
        dtype=float,
    )
    transform_step = Field[int](
        "Initial grid spacing in coadd pixels for approximating the transform "
        "from the tract frame to the detector frame.",
        default=25,
        dtype=int,
    )
    transform_edge_length = Field[float](
        "Maximum length in pixels for any polygon segments transformed "
        "between detector and tract coordinates.",
        default=50.0,
        dtype=float,
    )
    kernel_n_sigma = Field[float](
        "Truncation point for the Gaussian approximation to the PSF that is "
        "used as the warping/convolution kernel.",
        default=4.5,
        dtype=float,
    )
    bad_mask_planes = DictField[str, str](
        "Mask planes that mark pixels considered unusable by this algorithm. "
        "These pixels must still be populated with finite values (e.g. by "
        "interpolation).  Dictionary values are descriptions.",
        keytype=str,
        itemtype=str,
        # TODO[deferred]: check this list more thoroughly against real images
        default={
            "NO_DATA": "There was no data for this pixel in this detector.",
            "INTERPOLATED": "This pixel's value was interpolated prior to warping.",
            "CLIPPED": "An artifact was detected on this pixel when comparing warps.",
            "SATURATED": "This pixel was saturated.",
        },
    )
    psf_size = Field[int](
        "Size of warped/convolved PSF images (in both X and Y).",
        dtype=int,
        default=41,
    )
    n_noise_realizations = Field[int](
        "Number of noise realizations to generate and warp.",
        dtype=int,
        default=3,
    )

    def validate(self) -> None:
        super().validate()
        if "NO_DATA" not in self.bad_mask_planes:
            raise ValueError("The special NO_DATA plane must always be in bad_mask_planes.")


class GaussianKaiserWarpTask(Task):
    """A task for fused Gaussian-convolution/warping."""

    _DefaultName: ClassVar[str] = "warp"
    ConfigClass: ClassVar[type[GaussianKaiserWarpConfig]] = GaussianKaiserWarpConfig

    config: GaussianKaiserWarpConfig
    log: LsstLogAdapter

    def run(
        self,
        visit_image: VisitImage,
        patch: PatchDefinition,
        sky_projection: SkyProjection[TractFrame],
        band: str,
        rng_seed_base: int = 0,
    ) -> WarpedImage:
        """Warp a single-detector, single-visit image onto a coadd patch and
        convolve it with a per-cell Gaussian approximation to its PSF.

        Parameters
        ----------
        visit_image
            The input image to warp.
        patch
            The definition of the cell grid for PSF (and other) approximations,
            along with identifiers for this patch.
        sky_projection
            The coordinate projection from the target tract coordinate system
            to the sky.
        band
            Name of the bandpass filter.
        rng_seed_base
            Base random-number generator seed.  The index of the noise
            realization is added to this number to generate the full seed.

        Returns
        -------
        `WarpedImage`
            The warped and Gaussian-convolved patch-frame output image.

        Raises
        ------
        lsst.pipe.base.NoWorkFound
            Raised if the number of viable output pixels was too small.
        """
        detector_to_tract = visit_image.sky_projection.pixel_to_sky_transform.then(
            sky_projection.sky_to_pixel_transform
        )
        # The initial bounds here only detects cells that are missing due to
        # geometry, which isn't as precise as a pixel-level check. But it saves
        # us from wasting time on cells that definitely have no data, and it
        # guarantees that all of the cells we process are close enough to the
        # detector's region that we can trust its sky projection not to
        # extrapolate catastrophically.
        bounds = self._init_cell_grid_bounds(visit_image.bounds, patch, detector_to_tract)
        patch_warp = self._init_patch_warp(patch, bounds, sky_projection, band, visit_image.image.unit)
        for cell_index in bounds.cell_indices():
            self._warp_into_cell(
                patch_warp[bounds.grid.bbox_of(cell_index)],
                visit_image,
                detector_to_tract,
                cell_index=cell_index,
            )

    def _init_cell_grid_bounds(
        self,
        detector_bounds: Bounds,
        patch: PatchDefinition,
        detector_to_tract: Transform[DetectorFrame, TractFrame],
    ) -> CellGridBounds:
        if not isinstance(detector_bounds, Box | Polygon):
            detector_bounds = detector_bounds.bbox
        # It's important to start any geometric comparison here by
        # transforming the detector region to tract coordinates, because the
        # detector's sky projection might extrapolate badly but the simpler
        # tract sky projection can't.
        try:
            overlap_region = (
                detector_bounds.segmentize(self.config.transform_edge_length)
                .transform(detector_to_tract)
                .intersection(Polygon.from_box(patch.outer_bbox))
            )
        except NoOverlapError as err:
            raise NoWorkFound("No overlap between detector and patch outer bounding box.") from err
        overlap_region.prepare()
        missing = {
            cell_index
            for cell_index in patch.cells
            if not overlap_region.intersects(patch.cells.bbox_of(cell_index).to_polygon())
        }
        return CellGridBounds(grid=patch.cells, bbox=patch.outer_bbox, missing=frozenset(missing))

    def _init_patch_warp(
        self,
        patch: PatchDefinition,
        bounds: CellGridBounds,
        sky_projection: SkyProjection[TractFrame],
        band: str,
        unit: astropy.units.UnitBase,
    ) -> WarpedImage:
        psf_shape = bounds.subgrid_size.as_tuple() + (self.config.psf_size, self.config.psf_size)
        psf_array = np.zeros(psf_shape, dtype=np.float32)
        warp_kernel_array = np.zeros(psf_shape, dtype=np.float32)
        result = WarpedImage(
            Image(0.0, dtype=np.float32, bbox=patch.outer_bbox, unit=unit),
            mask_schema=MaskSchema([MaskPlane(k, v) for k, v in self.config.bad_mask_planes.items()]),
            sky_projection=sky_projection,
            band=band,
            psf=CellPointSpreadFunction(psf_array, bounds=bounds),
            warp_kernel=CellPointSpreadFunction(warp_kernel_array, bounds=bounds),
            patch=patch,
        )
        result.mask.set("NO_DATA")
        return result

    def _warp_into_cell(
        self,
        cell_warp: WarpedImage,
        visit_image: VisitImage,
        detector_to_tract: Transform[DetectorFrame, TractFrame],
        cell_index: CellIJ,
    ) -> bool:
        cell_on_detector: Polygon = cell_warp.bbox.segmentize(self.config.transform_edge_length).transform(
            detector_to_tract.inverted()
        )
        try:
            overlap = visit_image.bounds.intersection(cell_on_detector)
        except NoOverlapError:
            return False
        match overlap:
            case Box():
                overlap = overlap.to_polygon()
            case Polygon():
                pass
            case _:
                self.log.warning("Visit bounds intersection with cell %s is not a polygon.", cell_index)
                return False
        # Work out where we evaluate the PSF model.
        nominal_tract_eval_point = XY(
            x=math.floor(cell_warp.bbox.x.center), y=math.floor(cell_warp.bbox.y.center)
        )
        tract_eval_point: XY[int] = nominal_tract_eval_point
        detector_eval_point: XY[float]
        if overlap != cell_on_detector:
            # The detector boundary goes through this cell. Take the centroid
            # of the overlap polygon (in detector coordinates), transform that
            # to tract coordinates, round it to the nearest integer. We're
            # willing to assume that the PSF doesn't vary spatially over the
            # cell (that's fundamental to cell-based coaddition), but we want
            # to be very defensive about the subpixel phase we evaluate at,
            # because we know (from DES work) that keeping the coadd pixel
            # phase constant and averaging over the effectively random
            # observation pixel phases is okay for shear estimation (even
            # though we plan to mitigate phase effects in other ways now).
            tract_eval_point = detector_to_tract.apply_forward(overlap.centroid).map(round)
        detector_eval_point = detector_to_tract.apply_inverse(tract_eval_point)
        # Compute the input PSF image and its moments, which determine the
        # warping kernel.  We want the warp kernel to be separable for
        # performance reasons, so we ignore psf_ellipse.xy.
        try:
            psf_image = visit_image.psf.compute_stellar_image(
                x=detector_eval_point.x, y=detector_eval_point.y
            )
        except Exception:  # TODO[agent]: can we narrow this?
            self.log.warning("Could not evaluate PSF at %s for cell %s", detector_eval_point, cell_index)
            return False
        psf_ellipse = Ellipse.from_image_unweighted(
            psf_image, x=detector_eval_point.x, y=detector_eval_point.y
        ).remeasure_adaptive(psf_image)
        # We take the hull of the transformed cell bbox with the PSF image one
        # to guard against cases where the PSF is very large and/or was
        # evaluated close to the edge of the cell, and then we pad to the
        # kernel's support.  We only want to evaluate the coordinate transform
        # on a single grid, and this is the simplest way to do that.
        padded_input_bbox = Box.hull(overlap.bbox, psf_image.bbox).padded(
            self._kernel_support_half_size(psf_ellipse)
        )
        # TODO[perf]: would always_copy=True give us better cache locality
        # by keeping rows together?
        padded_input_image = visit_image.image.padded_to(padded_input_bbox)
        padded_input_mask = visit_image.mask.padded_to(padded_input_bbox, ["NO_DATA"])
        bad_bitmask = visit_image.mask.schema.bitmask(*self.config.bad_mask_planes)
        bad_pixels = (padded_input_mask.array & bad_bitmask).any(axis=-1)
        # TODO[deferred]: a better conceptual default for variance padding is
        # inf (both here and in the blank patch-level output image). But
        # non-finite values are viral, so we need to be very careful, not just
        # here but downstream as well.
        padded_input_variance = visit_image.variance.padded_to(padded_input_bbox, math.inf)
        padded_input_coverage = Image(
            1.0, dtype=np.float32, bbox=padded_input_bbox, sky_projection=visit_image.sky_projection
        )
        padded_input_coverage.array[bad_pixels] = 0
        padded_input_guess = Image(
            0.0, dtype=np.float32, bbox=padded_input_bbox, sky_projection=visit_image.sky_projection
        )
        padded_input_guess.array[bad_pixels] = padded_input_image.array[bad_pixels]
        # TODO[next]: generate noise realizations and noise_guess planes
        # Transform all of the points in the cell.
        xy_in: XY[np.ndarray] = detector_to_tract.apply_inverse_grid(
            cell_warp.bbox, in_atol=self.config.transform_atol, out_step=self.config.transform_step
        )

    def _kernel_support_half_size(self, ellipse: Ellipse) -> YX[int]:
        return YX(
            y=math.ceil(self.config.kernel_n_sigma * math.sqrt(ellipse.yy)),
            x=math.ceil(self.config.kernel_n_sigma * math.sqrt(ellipse.xx)),
        )


class _WarpWorkspace(BaseWarpImage):
    def __init__(
        self,
        image: Image,
        *,
        mask: Mask,
        variance: Image,
        coverage: Image,
        image_guess: Image,
        noise_realizations: Sequence[Image],
        noise_guesses: Sequence[Image],
        mask_schema: MaskSchema,
        sky_projection: SkyProjection[Any],
        psf: Image,
    ):
        super().__init__(
            image=image,
            mask=mask,
            variance=variance,
            coverage=coverage,
            image_guess=image_guess,
            noise_realizations=noise_realizations,
            noise_guesses=noise_guesses,
            mask_schema=mask_schema,
            sky_projection=sky_projection,
        )
