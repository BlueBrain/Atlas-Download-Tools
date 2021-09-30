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
"""Test for sync.py module."""
import numpy as np
import pytest
import responses
from PIL import Image

from atldld.sync import DatasetDownloader, get_parallel_transform, pir_to_xy, xy_to_pir


class TestGetParallelTransform:
    @pytest.mark.parametrize(
        "downsample_ref", [25, 50]
    )  # p, i, r are divisble by these
    def test_local_equals_API(self, pir_to_xy_response, downsample_ref):
        p = pir_to_xy_response["p"]
        i = pir_to_xy_response["i"]
        r = pir_to_xy_response["r"]

        axis = pir_to_xy_response["axis"]

        x = pir_to_xy_response["x"]
        y = pir_to_xy_response["y"]

        affine_2d = np.array(pir_to_xy_response["affine_2d"])
        affine_3d = np.array(pir_to_xy_response["affine_3d"])

        if axis == "coronal":
            assert i % downsample_ref == 0
            assert r % downsample_ref == 0

            slice_coordinate = p
            grid_shape = (8000 / downsample_ref, 11400 / downsample_ref)

        elif axis == "sagittal":
            assert p % downsample_ref == 0
            assert i % downsample_ref == 0

            slice_coordinate = r
            grid_shape = (13200 / downsample_ref, 8000 / downsample_ref)

        else:
            raise ValueError

        df = get_parallel_transform(
            slice_coordinate,
            affine_2d,
            affine_3d,
            axis=axis,
            downsample_ref=downsample_ref,  # the goal is to reduce computation
        )

        tx, ty = df.transformation

        assert tx.shape == grid_shape
        assert ty.shape == grid_shape

        # Asserts single pixel

        if axis == "coronal":
            i_ = int(i // downsample_ref)
            r_ = int(r // downsample_ref)

            x_pred, y_pred = tx[i_, r_], ty[i_, r_]
        elif axis == "sagittal":
            p_ = int(p // downsample_ref)
            i_ = int(i // downsample_ref)

            x_pred, y_pred = tx[p_, i_], ty[p_, i_]

        assert x_pred == pytest.approx(x, abs=1e-2)
        assert y_pred == pytest.approx(y, abs=1e-2)


class TestDatasetDownloader:
    @responses.activate
    def test_patched(self, data_folder, custom_cache_dir):
        """Does not requires internet, everything is patched."""

        # Parameters
        dataset_id = 123
        include_expression = False
        downsample_ref = 25
        axis = "sagittal"

        # Mocking the responses to request
        with open(data_folder / "sync" / "SectionDataSet.txt") as f:
            body_dataset = f.read()

        responses.add(
            responses.GET,
            "https://api.brain-map.org/api/v2/data/query.json?"
            "criteria=model::SectionDataSet,rma::criteria,[id$eq123],"
            "rma::include,alignment3d,rma::options[start_row$eq0][num_rows$eq25000]",
            body=body_dataset,
        )

        with open(data_folder / "sync" / "SectionImage.txt") as f:
            body_image = f.read()

        responses.add(
            responses.GET,
            "https://api.brain-map.org/api/v2/data/query.json?"
            "criteria=model::SectionImage,rma::criteria,"
            "%5Bdata_set_id$eq123%5D,rma::include,alignment2d,"
            "rma::options%5Bstart_row$eq0%5D%5Bnum_rows$eq25000%5D",
            body=body_image,
        )

        # Save fake image in cache folder
        custom_cache_dir.mkdir(exist_ok=True)
        fake_img = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))
        fake_img.save(custom_cache_dir / "101945191-0.jpg")

        # Call the function
        downloader = DatasetDownloader(
            dataset_id=dataset_id,
            include_expression=include_expression,
            downsample_ref=downsample_ref,
        )
        downloader.fetch_metadata()
        gen = downloader.run()

        if axis == "coronal":
            grid_shape = (8000 / downsample_ref, 11400 / downsample_ref)
        else:
            grid_shape = (13200 / downsample_ref, 8000 / downsample_ref)

        # Asserts - first iteration
        x = next(gen)
        assert len(x) == 5

        img_id, slice_coordinate, img, img_expression, df = x

        assert img_id == 101945191
        assert img_expression is None
        assert df.delta_x.shape == grid_shape
        assert df.delta_y.shape == grid_shape

        # Asserts remaining
        with pytest.raises(StopIteration):
            next(gen)


def test_pir_to_xy(pir_to_xy_response):
    coords_ref = np.array(
        [
            pir_to_xy_response["p"],
            pir_to_xy_response["i"],
            pir_to_xy_response["r"],
        ]
    )[
        :, None
    ]  # (3, 1)

    coords_img_API = np.array([pir_to_xy_response["x"], pir_to_xy_response["y"]])[
        :, None
    ]  # We do not care about the section coordinate

    coords_img_local = pir_to_xy(
        coords_ref,
        np.array(pir_to_xy_response["affine_2d"]),
        np.array(pir_to_xy_response["affine_3d"]),
    )[:2]

    assert np.allclose(coords_img_API, coords_img_local, rtol=0, atol=1e-3)


def test_xy_to_pir(xy_to_pir_response):
    coords_img = np.array(
        [
            xy_to_pir_response["x"],
            xy_to_pir_response["y"],
            xy_to_pir_response["section_number"]
            * xy_to_pir_response["section_thickness"],
        ]
    )[
        :, None
    ]  # (3, 1)

    coords_ref_API = np.array(
        [
            xy_to_pir_response["p"],
            xy_to_pir_response["i"],
            xy_to_pir_response["r"],
        ]
    )[:, None]

    coords_ref_local = xy_to_pir(
        coords_img,
        np.array(xy_to_pir_response["affine_2d"]),
        np.array(xy_to_pir_response["affine_3d"]),
    )

    assert np.allclose(coords_ref_API, coords_ref_local, rtol=0, atol=1e-3)
