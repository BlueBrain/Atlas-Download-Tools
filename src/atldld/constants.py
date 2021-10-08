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
"""Various constants."""

# Volume dimensions
REF_DIM_1UM = (13200, 8000, 11400)
REF_DIM_25UM = (528, 320, 456)

# Metadata retrieval
AFFINE_TEMPLATES = {
    "tsv": [
        ["tsv_00", "tsv_01", "tsv_04"],
        ["tsv_02", "tsv_03", "tsv_05"],
    ],
    "tvs": [
        ["tvs_00", "tvs_01", "tvs_04"],
        ["tvs_02", "tvs_03", "tvs_05"],
    ],
    "tvr": [
        ["tvr_00", "tvr_01", "tvr_02", "tvr_09"],
        ["tvr_03", "tvr_04", "tvr_05", "tvr_10"],
        ["tvr_06", "tvr_07", "tvr_08", "tvr_11"],
    ],
    "trv": [
        ["trv_00", "trv_01", "trv_02", "trv_09"],
        ["trv_03", "trv_04", "trv_05", "trv_10"],
        ["trv_06", "trv_07", "trv_08", "trv_11"],
    ],
}
