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
import enum


class PlaneOfSection(enum.Enum):
    CORONAL = 1
    SAGITTAL = 2

    def __str__(self):
        return self.name.lower()


class ReferenceSpace(enum.Enum):
    P56 = 9
    P56_LR_FLIPPED = 10

    __names__ = {
        P56: "P56 Brain",
        P56_LR_FLIPPED: "P56 Brain, L/R Flipped",
    }

    def __str__(self):
        return self.__names__[self.value]
