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
        assert "VARIABLE" in PlaneOfSection.__members__
        assert "NA" in PlaneOfSection.__members__
        assert "TRANSVERSE" in PlaneOfSection.__members__
        assert "LATERAL_TO_MEDIAL" in PlaneOfSection.__members__
        assert "MEDIAL_TO_LATERAL" in PlaneOfSection.__members__
        assert "VENTRAL_TO_DORSAL" in PlaneOfSection.__members__
        assert "DORSAL_TO_VENTRAL" in PlaneOfSection.__members__
        assert "CORONAL_CAUDAL_TO_ROSTRAL" in PlaneOfSection.__members__
        assert "CORONAL_ROSTRAL_TO_CAUDAL" in PlaneOfSection.__members__
        assert "TANGENTIAL" in PlaneOfSection.__members__

    def test_values(self):
        assert PlaneOfSection.CORONAL.value == 1
        assert PlaneOfSection.SAGITTAL.value == 2
        assert PlaneOfSection.VARIABLE.value == 4
        assert PlaneOfSection.NA.value == 5
        assert PlaneOfSection.TRANSVERSE.value == 6
        assert PlaneOfSection.LATERAL_TO_MEDIAL.value == 7
        assert PlaneOfSection.MEDIAL_TO_LATERAL.value == 8
        assert PlaneOfSection.VENTRAL_TO_DORSAL.value == 9
        assert PlaneOfSection.DORSAL_TO_VENTRAL.value == 10
        assert PlaneOfSection.CORONAL_CAUDAL_TO_ROSTRAL.value == 11
        assert PlaneOfSection.CORONAL_ROSTRAL_TO_CAUDAL.value == 531107579
        assert PlaneOfSection.TANGENTIAL.value == 1082279372

    def test_str(self):
        assert str(PlaneOfSection.CORONAL) == "coronal"
        assert str(PlaneOfSection.SAGITTAL) == "sagittal"
        assert str(PlaneOfSection.VARIABLE) == "variable"
        assert str(PlaneOfSection.NA) == "n/a"
        assert str(PlaneOfSection.TRANSVERSE) == "transverse"
        assert str(PlaneOfSection.LATERAL_TO_MEDIAL) == "Lateral to Medial"
        assert str(PlaneOfSection.MEDIAL_TO_LATERAL) == "Medial to Lateral"
        assert str(PlaneOfSection.VENTRAL_TO_DORSAL) == "Ventral to Dorsal"
        assert str(PlaneOfSection.DORSAL_TO_VENTRAL) == "Dorsal to Ventral"
        assert (
            str(PlaneOfSection.CORONAL_CAUDAL_TO_ROSTRAL)
            == "Coronal (Caudal to Rostral)"
        )
        assert (
            str(PlaneOfSection.CORONAL_ROSTRAL_TO_CAUDAL)
            == "Coronal (Rostral to Caudal)"
        )
        assert str(PlaneOfSection.TANGENTIAL) == "Tangential"


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
