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
import numpy as np
from matplotlib.figure import Figure

from atldld.dataset import PlaneOfSection
from atldld.plot import dataset_preview


class TestDatasetPreview:
    def test_coronal_dataset_works(self):
        # Coronal: horizontal axis = r, vertical axis = i
        all_corners = [
            np.array(
                [
                    [p, 0, 0],
                    [p, 0, 10000],
                    [p, 10000, 10000],
                    [p, 10000, 0],
                ]
            )
            for p in range(1000, 10000, 2000)
        ]
        fig = dataset_preview(all_corners, PlaneOfSection.CORONAL)
        assert isinstance(fig, Figure)
        # fig.savefig("test-dataset-preview-coronal.png")

    def test_sagittal_dataset_works(self):
        # Sagittal: horizontal axis = r, vertical axis = i
        all_corners = [
            np.array(
                [
                    [0, 0, r],
                    [10000, 0, r],
                    [10000, 10000, r],
                    [0, 10000, r],
                ]
            )
            for r in range(1000, 10000, 2000)
        ]
        fig = dataset_preview(all_corners, PlaneOfSection.SAGITTAL)
        assert isinstance(fig, Figure)
        # fig.savefig("test-dataset-preview-sagittal.png")
