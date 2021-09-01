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
from atldld.constants import REF_DIM_1UM, REF_DIM_25UM


def test_ref_space_dimensions_consistent():
    assert len(REF_DIM_1UM) == len(REF_DIM_25UM) == 3
    for dim_1um, dim_25um in zip(REF_DIM_1UM, REF_DIM_25UM):
        assert dim_25um == dim_1um / 25
