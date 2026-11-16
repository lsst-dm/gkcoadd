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

__all__ = ()

from typing import ClassVar

from lsst.images import SkyProjection, TractFrame, VisitImage
from lsst.images.cells import PatchDefinition
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from ._warped_image import WarpedImage


class GaussianKaiserWarpConfig(Config):
    """Configuration for GaussianKaiserWarpTask."""

    transform_atol = Field[float](
        "Absolute tolerance in coadd pixels for approximating the transform "
        "from the detector frame to the coadd frame.",
        default=0.005,
        dtype=float,
    )
    transform_step = Field[int](
        "Initial grid spacing in coadd pixels for approximating the transform "
        "from the detector frame to the coadd frame.",
        default=25,
        dtype=int,
    )
    kernel_n_sigma = Field[float](
        "Truncation point for the Gaussian approximation to the PSF that is "
        "used as the warping/convolution kernel.",
        default=3.0,
        dtype=float,
    )
    bad_pixel_mask = ListField[str](
        "Mask planes that mark pixels considered unusable by this algorithm. "
        "These pixels must still be populated with finite values (e.g. by "
        "interpolation).",
        # TODO[deferred]: check this list more thoroughly against real images
        default=["NO_DATA", "INTERPOLATED", "CLIPPED", "SATURATED"],
        dtype=str,
    )


class GaussianKaiserWarpTask(Task):
    """A task for fused Gaussian-convolution/warping."""

    _DefaultName: ClassVar[str] = "warp"
    ConfigClass: ClassVar[type[GaussianKaiserWarpConfig]] = GaussianKaiserWarpConfig

    def run(
        self, visit_image: VisitImage, patch: PatchDefinition, sky_projection: SkyProjection[TractFrame]
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

        Returns
        -------
        `WarpedImage`
            The warped and Gaussian-convolved patch-frame output image.

        Raises
        ------
        lsst.pipe.base.NoWorkFound
            Raised if the number of viable output pixels was too small.
        """
        raise NotImplementedError("TODO")
