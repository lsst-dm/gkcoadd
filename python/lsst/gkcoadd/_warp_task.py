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

import dataclasses
from collections.abc import Sequence

from lsst.images.psfs import PointSpreadFunction
from lsst.utils.logging import LsstLogAdapter

__all__ = ("GaussianKaiserWarpConfig", "GaussianKaiserWarpTask")

import math
from typing import Any, ClassVar, final

import astropy.units
import numpy as np
from astro_metadata_translator import ObservationInfo

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
    Region,
    SkyProjection,
    TractFrame,
    Transform,
    VisitImage,
)
from lsst.images.cells import CellGrid, CellIJ, PatchDefinition
from lsst.pex.config import Config, DictField, Field, ListField, RangeField
from lsst.pipe.base import InvalidQuantumError, NoWorkFound, Task, UpstreamFailureNoWorkFound

from ._cell_kernel import CellKernelWorkspace
from ._warped_image import BaseWarpImage, MissingCellReason, WarpedImage, WarpFields


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
    psf_image_size = Field[int](
        "Size of output warped/convolved PSF images (in both X and Y).",
        dtype=int,
        default=41,
    )
    psf_max_ellipticity = RangeField(
        "Maximum PSF ellipticity (reduced shear). "
        "PSFs with ellipticities larger than this are considered suspicious and their cells are dropped.",
        dtype=float,
        default=0.5,
        min=0.0,
        max=1.0,
        inclusiveMin=False,
        inclusiveMax=False,
    )
    psf_min_radius = Field[float](
        "Minimum PSF moment trace radius in detector pixels. "
        "PSFs smaller than this are considered suspicious and their cells are dropped.",
        dtype=float,
        default=1.0,
    )
    psf_max_radius = Field[float](
        "Maximum PSF moment trace radius in detector pixels. "
        "PSFs larger than this are expected to be truncated at psf_image_size and their cells are dropped.",
        dtype=float,
        default=15.0,
    )
    noise_realization_sources = ListField[str](
        "A list of variance sources to use for each noise realization. "
        "Values may be 'pixel' for the per-pixel variance plane or an ObservationSummaryStats "
        "('skyNoise' or 'meanVar' are currently supported) for constant (before calibration) "
        "per-detector noise.",
        dtype=str,
        default=["meanVar"] * 3,
    )
    noise_seed_offset = Field[int](
        "An offset included in the random number seed for all noise realizations, "
        "along with the visit and detector IDs.",
        dtype=int,
        default=0,
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

    def __init__(self, config: Config | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.mask_schema = MaskSchema([MaskPlane(k, v) for k, v in self.config.bad_mask_planes.items()])

    def run(
        self,
        *,
        visit_image: VisitImage,
        patch: PatchDefinition,
        sky_projection: SkyProjection[TractFrame],
        band: str,
        calibration_scaling: Image | None = None,
    ) -> WarpedImage:
        """Warp a single-detector, single-visit image onto a coadd patch and
        convolve it with a per-cell Gaussian approximation to its PSF.

        Parameters
        ----------
        visit_image
            The input image to warp.  Expected to already have calibrated
            pixel units.
        patch
            The definition of the cell grid for PSF (and other) approximations,
            along with identifiers for this patch.
        sky_projection
            The coordinate projection from the target tract coordinate system
            to the sky.
        band
            Name of the bandpass filter.
        calibration_scaling
            A rendered image of the mapping from instrumental units (as used
            by `lsst.images.ObservationSummaryStats` fields) to the visit
            image's calibrated units.  Computed on first use if not provided.

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
        workspace = _PatchWarpWorkspace.make_empty(
            patch,
            sky_projection,
            len(self.config.noise_realization_sources),
            mask_schema=self.mask_schema,
            psf_shape=YX(y=self.config.psf_image_size, x=self.config.psf_image_size),
        )
        overlap_region = self._compute_coarse_patch_overlap(visit_image.bounds, patch, detector_to_tract)
        # Noise realizations cover the full detector and are generated up
        # front, so cells and patches sharing a detector share the same noise.
        noise_realizations = self._generate_noise(visit_image, calibration_scaling=calibration_scaling)
        for cell_index in patch.cells:
            cell_bbox = patch.cells.bbox_of(cell_index)
            if not overlap_region.intersects(cell_bbox.to_polygon()):
                # The 'empty' state of a workspace cell already matches this
                # condition.
                continue
            try:
                self._warp_into_cell(
                    workspace[cell_index],
                    visit_image,
                    noise_realizations,
                    detector_to_tract,
                    cell_index=cell_index,
                )
            except _SkipCellError as err:
                workspace.missing[cell_index] = err.reason
            else:
                del workspace.missing[cell_index]
        if len(workspace.missing) == len(workspace.grid):
            raise NoWorkFound("No viable cells in warp.")
        return workspace.finish(visit_image.obs_info, band, patch)

    def _compute_coarse_patch_overlap(
        self,
        detector_bounds: Bounds,
        patch: PatchDefinition,
        detector_to_tract: Transform[DetectorFrame, TractFrame],
    ) -> Region:
        """Compute the coarse overlap between the detector-level input image
        and the patch, as a coadd-coordinate region.
        """
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
        return overlap_region

    def _generate_noise(
        self, visit_image: VisitImage, calibration_scaling: Image | None = None
    ) -> list[Image]:
        """Generate full-detector noise images.

        Parameters
        ----------
        visit_image
            Image whose projection, units, and noise properties the noise
            images should reflect.
        calibration_scaling
            A rendered image of the mapping from instrumental units (as used
            by `lsst.images.ObservationSummaryStats` fields) to the visit
            image's calibrated units.  Computed on first use if not provided.
        """
        result: list[Image] = []
        if (visit_id := visit_image.obs_info.visit_id) is None:
            raise InvalidQuantumError("ObservationInfo.visit_id is not populated.")
        if (detector_num := visit_image.obs_info.detector_num) is None:
            raise InvalidQuantumError("ObservationInfo.detector_num is not populated.")
        rng_seed_seq = np.random.SeedSequence(entropy=[visit_id, detector_num, self.config.noise_seed_offset])
        for rng_seed, noise_source in zip(
            rng_seed_seq.spawn(len(self.config.noise_realization_sources)),
            self.config.noise_realization_sources,
        ):
            rng = np.random.default_rng(rng_seed)
            is_calibrated = False
            match noise_source:
                case "pixel":
                    scale = visit_image.variance.array.astype(np.float32, copy=False) ** 0.5
                    array = rng.standard_normal(size=scale.shape, dtype=np.float32) * scale
                    is_calibrated = True
                case "meanVar":
                    array = rng.standard_normal(size=visit_image.bbox.shape, dtype=np.float32) * np.float32(
                        visit_image.summary_stats.meanVar**0.5
                    )
                case "skyNoise":
                    array = rng.standard_normal(size=visit_image.bbox.shape, dtype=np.float32) * np.float32(
                        visit_image.summary_stats.skyNoise
                    )
                case _:
                    raise ValueError(f"Unknown noise source: {noise_source!r}.")
            if not is_calibrated:
                if calibration_scaling is None:
                    if visit_image.photometric_scaling is None:
                        raise UpstreamFailureNoWorkFound("Visit image has no photometric calibration.")
                    calibration_scaling = visit_image.photometric_scaling.render(
                        visit_image.bbox, dtype=np.float32
                    )
                array *= calibration_scaling.array
            result.append(
                Image(
                    array,
                    dtype=np.float32,
                    bbox=visit_image.bbox,
                    unit=visit_image.unit,
                    sky_projection=visit_image.sky_projection,
                )
            )
        return result

    def _warp_into_cell(
        self,
        output_workspace: _SingleCellWarpWorkspace,
        visit_image: VisitImage,
        noise_realizations: list[Image],
        detector_to_tract: Transform[DetectorFrame, TractFrame],
        cell_index: CellIJ,
    ) -> None:
        overlap, contained = self._compute_cell_overlap(
            output_workspace.bbox, visit_image.bounds, detector_to_tract, cell_index
        )
        # Work out where we evaluate the PSF model.
        tract_eval_point = XY(
            x=math.floor(output_workspace.bbox.x.center), y=math.floor(output_workspace.bbox.y.center)
        )
        if not contained:
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
        # Evaluate and measure the PSF model.
        psf_image, psf_ellipse = self._process_cell_input_psf(
            visit_image.psf,
            detector_to_tract.apply_inverse(tract_eval_point.map(float)),
            cell_index,
        )
        # We take the hull of the transformed cell bbox with the PSF image one
        # to guard against cases where the PSF is very large and/or was
        # evaluated close to the edge of the cell, and then we pad to the
        # kernel's support. We only want to evaluate the coordinate transform
        # on a single grid when possible, and this is the simplest way to do
        # that.
        padded_input_bbox = Box.hull(overlap.bbox, psf_image.bbox).padded(
            self._kernel_support_half_size(psf_ellipse)
        )
        input_workspace = _SingleCellWarpWorkspace.from_visit_image(
            visit_image,
            psf_image,
            bbox=padded_input_bbox,
            bad_mask_planes=list(self.config.bad_mask_planes.keys()),
            noise_realizations=noise_realizations,
        )
        self._render_warp_kernel(input_workspace.warp_kernel, psf_ellipse)
        # Transform all of the points in the cell.
        image_xy, kernel_xy = self._transform_grids(
            image_bbox=output_workspace.bbox,
            kernel_bbox=output_workspace.psf.bbox,
            tract_eval_point=tract_eval_point,
            detector_to_tract=detector_to_tract,
        )
        return self._do_warp(output_workspace, input_workspace, psf_ellipse, image_xy, kernel_xy)

    def _compute_cell_overlap(
        self,
        output_bbox: Box,
        input_bounds: Bounds,
        detector_to_tract: Transform[DetectorFrame, TractFrame],
        cell_index: CellIJ,
    ) -> tuple[Polygon, bool]:
        """Compute the overlap between a cell and the input visit image.

        Parameters
        ----------
        output_bbox
            The output cell bounding box, in tract coordinates.
        input_bounds
            The bounds of the input visit image; typically (but not
            necessarily) just its bounding box.
        detector_to_tract
            Coordinate transform from detector coordinates to tract
            coordinates.
        cell_index
            Index of the cell being processed.

        Returns
        -------
        overlap : `lsst.images.Polygon`
            The overlap polygon in detector coordinates.
        contained : `bool`
            `True` if the cell lies wholly within the detector, `False` if
            it's on the boundary.
        """
        cell_on_detector: Polygon = output_bbox.segmentize(self.config.transform_edge_length).transform(
            detector_to_tract.inverted()
        )
        try:
            overlap = input_bounds.intersection(cell_on_detector)
        except NoOverlapError:
            return MissingCellReason.NO_COARSE_OVERLAP
        match overlap:
            case Box():
                overlap = overlap.to_polygon()
            case Polygon():
                pass
            case _:
                self.log.warning("Visit bounds intersection with cell %s is not a polygon.", cell_index)
                raise _SkipCellError(MissingCellReason.UNEXPECTED_OVERLAP_GEOMETRY)
        return overlap, overlap == cell_on_detector

    def _process_cell_input_psf(
        self,
        input_psf: PointSpreadFunction,
        eval_point: XY[float],
        cell_index: CellIJ,
    ) -> tuple[Image, Ellipse]:
        """Evaluate and measure the moments of the input PSF for a cell.

        Parameters
        ----------
        input_psf
            The PSF model to evaluate.
        eval_point
            Where to evaluate the PSF, in detector coordinates.
        cell_index
            Index of the cell being processed.

        Returns
        -------
        psf_image : `lsst.images.Image`
            Image of the PSF, in detector coordinates.
        psf_ellipse : `lsst.images.Ellipse`
            Second-moments ellipse of the PSF model at the eval point.
        """
        # Compute the input PSF image and its moments, which determine the
        # warping kernel.  We want the warp kernel to be separable for
        # performance reasons, so we ignore psf_ellipse.xy.
        try:
            psf_image = input_psf.compute_stellar_image(x=eval_point.x, y=eval_point.y)
        except (ValueError, RuntimeError):
            # PSF backends signal out-of-domain points with pex (and GalSim)
            # RuntimeErrors; some raise ValueError.
            self.log.warning("Could not evaluate PSF at %s for cell %s.", eval_point, cell_index)
            raise _SkipCellError(MissingCellReason.FAILED_PSF_EVALUATION)
        try:
            psf_ellipse = Ellipse.from_image_unweighted(
                psf_image, x=eval_point.x, y=eval_point.y
            ).remeasure_adaptive(psf_image)
        except (ValueError, RuntimeError):
            # Ellipse construction raises InvalidEllipseError (a ValueError);
            # GalSim's HSM raises RuntimeError subclasses.
            self.log.warning("Could not measure PSF moments at %s for cell %s.", eval_point, cell_index)
            raise _SkipCellError(MissingCellReason.FAILED_PSF_MOMENTS)
        if abs(psf_ellipse.reduced_shear) > self.config.psf_max_ellipticity:
            self.log.warning(
                "PSF ellipticity %f > %f at %s for cell %s.",
                abs(psf_ellipse.reduced_shear),
                self.config.psf_max_ellipticity,
                eval_point,
                cell_index,
            )
            raise _SkipCellError(MissingCellReason.FAILED_PSF_MOMENTS)
        if psf_ellipse.r_tr < self.config.psf_min_radius:
            self.log.warning(
                "PSF radius %f < %f at %s for cell %s.",
                psf_ellipse.r_tr,
                self.config.psf_min_radius,
                eval_point,
                cell_index,
            )
            raise _SkipCellError(MissingCellReason.FAILED_PSF_MOMENTS)
        if psf_ellipse.r_tr > self.config.psf_max_radius:
            self.log.info(
                "PSF radius %f > %f at %s for cell %s.",
                psf_ellipse.r_tr,
                self.config.psf_max_radius,
                eval_point,
                cell_index,
            )
            raise _SkipCellError(MissingCellReason.PSF_TOO_LARGE)
        return psf_image, psf_ellipse

    def _kernel_support_half_size(self, ellipse: Ellipse) -> YX[int]:
        """Return the half-size of the kernel's support box.

        Parameters
        ----------
        ellipse
            PSF second-moments ellipse.

        Returns
        -------
        `lsst.images.YX` [`int`]
            Half-size of the kernel's support box.  The support box shape
            is twice this *plus one*, since it always has odd dimensions.
        """
        return YX(
            y=math.ceil(self.config.kernel_n_sigma * math.sqrt(ellipse.yy)),
            x=math.ceil(self.config.kernel_n_sigma * math.sqrt(ellipse.xx)),
        )

    def _render_warp_kernel(self, image: Image, psf_ellipse: Ellipse) -> None:
        """Render the PSF-approximating Gaussian warp kernel.

        Parameters
        ----------
        image
            Image to write into.
        psf_ellipse
            PSF second-moments ellipse.

        Notes
        -----
        The `~lsst.images.Ellipse.xy` cross-moment is ignored, since it's more
        important for this kernel to be separable than it is for it to
        approximate the PSF closely.
        """
        g = image.bbox.meshgrid()
        g.x[...] -= psf_ellipse.center.x
        g.y[...] -= psf_ellipse.center.y
        np.square(g.x, out=g.x)
        np.square(g.y, out=g.y)
        arg = g.x / psf_ellipse.xx + g.y / psf_ellipse.yy
        # arg is squared distance in sigma units, so truncate at n_sigma**2.
        mask = arg < self.config.kernel_n_sigma**2
        image.array[mask] = np.exp(-0.5 * arg[mask])
        image.array /= image.array.sum()

    def _transform_grids(
        self,
        image_bbox: Box,
        kernel_bbox: Box,
        tract_eval_point: XY[int],
        detector_to_tract: Transform[DetectorFrame, TractFrame],
    ) -> tuple[XY[np.ndarray], XY[np.ndarray]]:
        """Compute the input pixel coordinates on the image and kernel
        output grids.

        Parameters
        ----------
        image_bbox
            Bounding box of the tract-coordinate regular output image planes.
        kernel_bbox
            Bounding box of the output kernel image planes, centered on zero.
        tract_eval_point
            The tract coordinate that corresponds to (0, 0) in ``kernel_bbox``.
        detector_to_tract
            Coordinate transform from detector coordinates to tract
            coordinates.

        Returns
        -------
        image_xy : `XY` [`numpy.ndarray`]
            Grid of detector-coordinate values for the output image planes.
        kernel_xy : `XY` [`numpy.ndarray`]
            Grid of detector-coordinate values for the output PSF and
            warp kernel planes.
        """
        image_xy: XY[np.ndarray] = detector_to_tract.apply_inverse_grid(
            image_bbox, in_atol=self.config.transform_atol, out_step=self.config.transform_step
        )
        # The PSF and warp_kernel input images are in regular detector
        # coordinates, but we want to warp those to the
        # centered_kernel_bbox coordinate system where tract_eval_point
        # becomes (0, 0).
        tract_kernel_bbox = Box.factory[
            tract_eval_point.y + kernel_bbox.y.start : tract_eval_point.y + kernel_bbox.y.stop,
            tract_eval_point.x + kernel_bbox.x.start : tract_eval_point.x + kernel_bbox.x.stop,
        ]
        if image_bbox.contains(tract_kernel_bbox):
            # In the usual case, kernel output box is fully within the cell,
            # so we've already computed its input-frame coordinate arrays.
            # YX[slice] indexes ndarray members directly (it is a tuple).
            kernel_slices = tract_kernel_bbox.slice_within(image_bbox)
            kernel_xy = XY(x=image_xy.x[kernel_slices], y=image_xy.y[kernel_slices])
        else:
            # When the kernel is too close to the edge of the cell, we need
            # to transform new points.  This is rare enough that we don't
            # bother with partial reuse of already-transformed points.
            kernel_xy = detector_to_tract.apply_inverse_grid(
                tract_kernel_bbox, in_atol=self.config.transform_atol, out_step=self.config.transform_step
            )
        return image_xy, kernel_xy

    def _do_warp(
        self,
        output_workspace: _SingleCellWarpWorkspace,
        input_workspace: _SingleCellWarpWorkspace,
        kernel_ellipse: Ellipse,
        image_xy: XY[np.ndarray],
        kernel_xy: XY[np.ndarray],
    ) -> MissingCellReason | None:
        """Actually do the warping.

        Parameters
        ----------
        output_workspace
            Output warped image cell to write into.
        input_workspace
            Input padded-as-needed visit image and synthetic planes to warp
            from.  Guaranteed to have full kernel support over every output
            pixel.
        kernel_ellipse
            The ellipse that defines the Gaussian warping kernel.
        image_xy
            Arrays with the input pixel coordinate value for every image
            output pixel; shapes are the same as
            ``output_workspace.bbox.shape``.
        kernel_xy
            Arrays with the input pixel coordinate values for ``psf``
            and ``warp_kernel`` output pixels; shapes are the same as those
            output images.
        """
        raise NotImplementedError("TODO")


class _SkipCellError(RuntimeError):
    """Exception raised when a warp cell should be skipped."""

    def __init__(self, reason: MissingCellReason):
        self.reason = reason


@final
class _SingleCellWarpWorkspace(BaseWarpImage[Image]):
    @classmethod
    def from_visit_image(
        cls,
        visit_image: VisitImage,
        psf_image: Image,
        bbox: Box,
        bad_mask_planes: Sequence[str],
        noise_realizations: Sequence[Image],
    ) -> _SingleCellWarpWorkspace:
        # 'Any' here silences MyPy grumpiness about using **kwargs.
        common_kwargs: Any = {"dtype": np.float32, "bbox": bbox, "sky_projection": visit_image.sky_projection}
        result = cls(
            {
                "image": visit_image.image.padded_to(bbox),
                # always_copy: the NO_DATA marking below must not write
                # through into the visit image when bbox fits inside it.
                "mask": visit_image.mask.padded_to(bbox, ["NO_DATA"], always_copy=True),
                # TODO: 'inf' as a fill value is problematic here, because the
                # kernel will spread it around.  We need a scheme for using
                # 'coverage' to inflate variances in the coadd sum instead.
                # I hope that doesn't require a variance-coverage plane.
                "variance": visit_image.variance.padded_to(bbox, math.inf),
                "coverage": Image(1.0, unit=None, **common_kwargs),
                "image_guess": Image(0.0, unit=visit_image.unit, **common_kwargs),
                "noise_realizations": [r.padded_to(bbox) for r in noise_realizations],
                "noise_guesses": [
                    Image(0.0, unit=visit_image.unit, **common_kwargs) for _ in noise_realizations
                ],
                "psf": psf_image.padded_to(bbox).view(sky_projection=visit_image.sky_projection),
                "warp_kernel": Image(0.0, unit=None, **common_kwargs),
            }
        )
        # Pixels outside the bounds should generally be marked NO_DATA
        # already, but make sure; no bbox fast path, since it would miss
        # pixels outside a polygonal bounds.
        out_of_bounds = np.logical_not(visit_image.bounds.contains(bbox.meshgrid()))
        result.mask.set("NO_DATA", out_of_bounds)
        bad_bitmask = result.mask.schema.bitmask(*bad_mask_planes)
        bad_pixels = (result.mask.array & bad_bitmask).any(axis=-1)
        result.coverage.array[bad_pixels] = 0
        result.image_guess.array[bad_pixels] = result.image.array[bad_pixels]
        for r, g in zip(result.noise_realizations, result.noise_guesses):
            g.array[bad_pixels] = r.array[bad_pixels]
        return result

    def _make_modified(self, fields: WarpFields) -> _SingleCellWarpWorkspace:
        return _SingleCellWarpWorkspace(fields)


@final
class _PatchWarpWorkspace(BaseWarpImage[CellKernelWorkspace]):
    """An under-construction warped image covering a full patch."""

    def __init__(self, fields: WarpFields, missing: dict[CellIJ, MissingCellReason]):
        super().__init__(fields)
        self._missing = missing

    @classmethod
    def make_empty(
        cls,
        patch: PatchDefinition,
        sky_projection: SkyProjection[TractFrame],
        n_noise_realizations: int,
        mask_schema: MaskSchema,
        psf_shape: YX[int],
    ) -> _PatchWarpWorkspace:
        # 'Any' here silences MyPy grumpiness about using **kwargs.
        common_kwargs: Any = {"bbox": patch.outer_bbox, "sky_projection": sky_projection, "dtype": np.float32}
        return cls(
            {
                "image": Image(0.0, unit=astropy.units.nJy, **common_kwargs),
                "mask": Mask(
                    mask_schema.bitmask("NO_DATA"),
                    schema=mask_schema,
                    bbox=patch.outer_bbox,
                    sky_projection=sky_projection,
                ),
                "variance": Image(np.inf, unit=astropy.units.nJy**2, **common_kwargs),
                "coverage": Image(0.0, unit=None, **common_kwargs),
                "image_guess": Image(0.0, unit=astropy.units.nJy, **common_kwargs),
                "noise_realizations": [
                    Image(0.0, unit=astropy.units.nJy, **common_kwargs) for _ in range(n_noise_realizations)
                ],
                "noise_guesses": [
                    Image(0.0, unit=astropy.units.nJy, **common_kwargs) for _ in range(n_noise_realizations)
                ],
                "psf": CellKernelWorkspace.make_empty(psf_shape, patch.cells),
                "warp_kernel": CellKernelWorkspace.make_empty(psf_shape, patch.cells),
            },
            missing=dict.fromkeys(patch.cells, MissingCellReason.NO_COARSE_OVERLAP),
        )

    @property
    def grid(self) -> CellGrid:
        """The patch-level grid of cells (`lsst.images.cells.CellGrid`)."""
        return self._psf.grid

    @property
    def missing(self) -> dict[CellIJ, MissingCellReason]:
        """The cells that do not have any data yet, and why
        (`dict` [`lsst.images.cells.CellIJ`, `MissingCellReason`]).

        Notes
        -----
        This is initialized with all cells missing, and must be updated in
        concert with the image and kernel fields manually; no consistency is
        maintained by this class.
        """
        return self._missing

    # Not accepting Box or EllipsisType here is a deliberate Liskov violation;
    # it's a smaller price to pay than not inheriting from MaskedImage or
    # implementing Box slicing that nothing would use.
    def __getitem__(self, index: CellIJ) -> _SingleCellWarpWorkspace:  # type: ignore[override]
        bbox = self.grid.bbox_of(index)
        # We use Any because we're switching in place from
        # WarpFields[CellKernelWorkspace] to WarpFields[Image] via the
        # assignments below.
        fields: WarpFields[Any] = self._slice_image_fields(bbox)
        fields["psf"] = self.psf[index]
        fields["warp_kernel"] = self.warp_kernel[index]
        return _SingleCellWarpWorkspace(fields)

    def finish(self, obs_info: ObservationInfo, band: str, patch: PatchDefinition) -> WarpedImage:
        """Finish building the warp, converting this object into a
        `WarpedImage`.
        """
        return WarpedImage(
            {
                "image": self.image,
                "mask": self.mask,
                "variance": self.variance,
                "coverage": self.coverage,
                "image_guess": self.image_guess,
                "noise_realizations": list(self.noise_realizations),
                "noise_guesses": list(self.noise_guesses),
                "psf": self.psf.finish(self.missing.keys()),
                "warp_kernel": self.warp_kernel.finish(self.missing.keys()),
            },
            obs_info=obs_info,
            band=band,
            patch=patch,
        )

    def _make_modified(self, fields: WarpFields) -> _PatchWarpWorkspace:
        # We copy self.missing because super().copy() is the only way to get
        # here; __getitem__ doesn't support bbox slicing.
        return _PatchWarpWorkspace(fields, self.missing.copy())
