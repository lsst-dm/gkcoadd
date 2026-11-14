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

"""Smoke tests for the package itself."""

import lsst.gkcoadd


def test_version() -> None:
    """Test that the version generated at build time is usable.

    sconsUtils writes a bare commit hash (plus a ``-dirty`` suffix) for
    untagged builds, so this cannot be checked as a PEP 440 version; only the
    versions lsst-versions generates for a wheel are.
    """
    assert isinstance(lsst.gkcoadd.__version__, str)
    assert lsst.gkcoadd.__version__
