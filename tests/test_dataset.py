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
from atldld.dataset import PlaneOfSection, ReferenceSpace


class TestPlaneOfSection:
    def test_members(self):
        assert "CORONAL" in PlaneOfSection.__members__
        assert "SAGITTAL" in PlaneOfSection.__members__

    def test_values(self):
        assert PlaneOfSection.CORONAL.value == 1
        assert PlaneOfSection.SAGITTAL.value == 2

    def test_str(self):
        assert str(PlaneOfSection.CORONAL) == "coronal"
        assert str(PlaneOfSection.SAGITTAL) == "sagittal"


class TestReferenceSpace:
    def test_members(self):
        assert "P56" in ReferenceSpace.__members__
        assert "P56_LR_FLIPPED" in ReferenceSpace.__members__

    def test_values(self):
        assert ReferenceSpace.P56.value == 9
        assert ReferenceSpace.P56_LR_FLIPPED.value == 10

    def test_str(self):
        assert str(ReferenceSpace.P56) == "P56 Brain"
        assert str(ReferenceSpace.P56_LR_FLIPPED) == "P56 Brain, L/R Flipped"
