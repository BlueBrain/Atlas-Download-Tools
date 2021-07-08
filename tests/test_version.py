# The package atldld is a tool to download atlas data.
#
# Copyright (C) 2021 EPFL/Blue Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import atldld


def test_version_exists():
    # Version exists
    assert hasattr(atldld, "__version__")
    assert isinstance(atldld.__version__, str)
    parts = atldld.__version__.split(".")

    # Version has correct format
    # Can be either "X.X.X" or "X.X.X.devX"
    assert len(parts) in {3, 4}
    assert parts[0].isdecimal()  # major
    assert parts[1].isdecimal()  # minor
    assert parts[1].isdecimal()  # patch
