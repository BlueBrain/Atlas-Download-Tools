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
