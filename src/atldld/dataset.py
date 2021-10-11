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
"""Everything related to section image datasets."""
import enum


class PlaneOfSection(enum.Enum):
    """The plane of section in a section image dataset.

    The values correspond to the IDs used by the AIBS and can be found here:
    http://api.brain-map.org/api/v2/data/PlaneOfSection/query.json
    """

    CORONAL = 1
    SAGITTAL = 2
    VARIABLE = 4
    NA = 5
    TRANSVERSE = 6
    LATERAL_TO_MEDIAL = 7
    MEDIAL_TO_LATERAL = 8
    VENTRAL_TO_DORSAL = 9
    DORSAL_TO_VENTRAL = 10
    CORONAL_CAUDAL_TO_ROSTRAL = 11
    CORONAL_ROSTRAL_TO_CAUDAL = 531107579
    TANGENTIAL = 1082279372

    __names__ = {
        CORONAL: "coronal",
        SAGITTAL: "sagittal",
        VARIABLE: "variable",
        NA: "n/a",
        TRANSVERSE: "transverse",
        LATERAL_TO_MEDIAL: "Lateral to Medial",
        MEDIAL_TO_LATERAL: "Medial to Lateral",
        VENTRAL_TO_DORSAL: "Ventral to Dorsal",
        DORSAL_TO_VENTRAL: "Dorsal to Ventral",
        CORONAL_CAUDAL_TO_ROSTRAL: "Coronal (Caudal to Rostral)",
        CORONAL_ROSTRAL_TO_CAUDAL: "Coronal (Rostral to Caudal)",
        TANGENTIAL: "Tangential",
    }

    def __str__(self):
        """Get the name of the plane of section."""
        return self.__names__[self.value]


class ReferenceSpace(enum.Enum):
    """The reference space of a section image.

    The values correspond to the IDs used by the AIBS and can be found here:
    http://api.brain-map.org/api/v2/data/ReferenceSpace/query.json

    Only a subset of values of interest are included. For the full list see the
    URL above.
    """

    P56 = 9
    P56_LR_FLIPPED = 10

    __names__ = {
        P56: "P56 Brain",
        P56_LR_FLIPPED: "P56 Brain, L/R Flipped",
    }

    def __str__(self):
        """Get the the name of the reference space."""
        return self.__names__[self.value]
