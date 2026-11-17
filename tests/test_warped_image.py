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

"""Tests for WarpedImage construction, including the checks and defaults for
the additional image planes.
"""

import astropy.units as u
import astropy.wcs
import numpy as np
import pytest
from astropy.io import fits

from lsst.images import Box, Image, MaskPlane, MaskSchema, SkyProjection, TractFrame, YX
from lsst.images.cells import CellGrid, CellGridBounds, CellPointSpreadFunction

from lsst.gkcoadd import WarpedImage

BBOX = Box.from_shape((64, 64))

SUB_BBOX = Box.factory[0:32, 0:32]

SKY_PROJECTION = SkyProjection.from_fits_wcs(
    astropy.wcs.WCS(
        fits.Header(
            {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 32.5,
                "CRPIX2": 32.5,
                "CRVAL1": 42.0,
                "CRVAL2": -15.0,
                "CD1_1": -2e-5,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": 2e-5,
            }
        )
    ),
    TractFrame(skymap="unknown", tract=1, bbox=Box.from_shape((2048, 2048))),
)

CELL_BOUNDS = CellGridBounds(
    grid=CellGrid(bbox=BBOX, cell_shape=YX(y=32, x=32)),
    bbox=BBOX,
)

PSF = CellPointSpreadFunction(np.ones((2, 2, 9, 9), dtype=np.float32), bounds=CELL_BOUNDS)

MASK_SCHEMA = MaskSchema([MaskPlane("NO_DATA", "There was no data for this pixel.")])


def make_image(value: float = 1.0, unit: u.UnitBase | None = u.nJy, **kwargs) -> Image:
    """Make an image sized and projected like the test warp's main image."""
    return Image(
        np.full(BBOX.shape, value, dtype=np.float32),
        unit=unit,
        sky_projection=SKY_PROJECTION,
        **kwargs,
    )


def make_warped(**kwargs) -> WarpedImage:
    """Make a minimal WarpedImage, overriding any of the standard components."""
    kwargs.setdefault("mask_schema", MASK_SCHEMA)
    return WarpedImage(
        make_image(),
        psf=PSF,
        warp_kernel=PSF,
        **kwargs,
    )


class TestDefaults:
    def test_coverage_defaults_to_zeros(self):
        coverage = make_warped()._coverage
        assert coverage.bbox == BBOX
        assert coverage.unit is None
        assert coverage.sky_projection is SKY_PROJECTION
        np.testing.assert_array_equal(coverage.array, 0)

    def test_image_guess_defaults_to_zeros_in_image_units(self):
        image_guess = make_warped()._image_guess
        assert image_guess.bbox == BBOX
        assert image_guess.unit == u.nJy
        assert image_guess.sky_projection is SKY_PROJECTION
        np.testing.assert_array_equal(image_guess.array, 0)

    def test_noise_guesses_default_to_zeros(self):
        warped = make_warped(noise_realizations=[make_image(), make_image()])
        assert len(warped.noise_realizations) == 2
        assert len(warped._noise_guesses) == 2
        for guess in warped._noise_guesses:
            assert guess.unit == u.nJy
            np.testing.assert_array_equal(guess.array, 0)


class TestValidation:
    def test_bbox_mismatch_rejected(self):
        with pytest.raises(ValueError, match="bboxes do not agree"):
            make_warped(coverage=Image(np.zeros((16, 16), dtype=np.float32)))

    def test_unit_mismatch_rejected(self):
        with pytest.raises(ValueError, match="coverage must be dimensionless"):
            make_warped(coverage=make_image())
        with pytest.raises(ValueError, match="image_guess unit"):
            make_warped(image_guess=make_image(unit=u.Jy))

    def test_unit_none_adopted_from_image(self):
        image_guess = make_warped(image_guess=make_image(unit=None))._image_guess
        assert image_guess.unit == u.nJy

    def test_dimensionless_coverage_accepted(self):
        coverage = make_warped(coverage=make_image(unit=u.dimensionless_unscaled))._coverage
        assert coverage.unit == u.dimensionless_unscaled

    def test_projection_replaced(self):
        plane = Image(np.ones(BBOX.shape, dtype=np.float32), unit=u.nJy)
        warped = make_warped(image_guess=plane)
        assert warped._image_guess.sky_projection is SKY_PROJECTION

    def test_noise_guesses_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="same length as noise_realizations"):
            make_warped(noise_realizations=[make_image(), make_image()], noise_guesses=[make_image()])


class TestDerivedImages:
    def test_getitem(self):
        warped = make_warped(
            coverage=make_image(value=0.5, unit=None),
            noise_realizations=[make_image()],
        )
        subset = warped[SUB_BBOX]
        assert subset.bbox == SUB_BBOX
        assert subset._coverage.bbox == SUB_BBOX
        np.testing.assert_array_equal(subset._coverage.array, 0.5)
        assert len(subset._noise_guesses) == 1

    def test_copy(self):
        warped = make_warped(coverage=make_image(value=0.5, unit=None))
        duplicate = warped.copy()
        np.testing.assert_array_equal(duplicate._coverage.array, 0.5)
        duplicate._coverage.array[...] = 0.0
        np.testing.assert_array_equal(warped._coverage.array, 0.5)
