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
from unittest.mock import Mock

import numpy as np
import pytest

from atldld.sync import (
    download_parallel_dataset,
    get_parallel_transform,
    pir_to_xy,
    xy_to_pir,
)


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


class TestDownloadParallelDataset:
    @pytest.mark.parametrize("include_expression", [True, False])
    @pytest.mark.parametrize("downsample_ref", [25, 50])
    @pytest.mark.parametrize("axis", ["coronal", "sagittal"])
    def test_patched(self, include_expression, downsample_ref, axis, monkeypatch):
        """Does not requires internet, everything is patched.

        The only thing that is unpatched is the `get_transform_simple`.
        """

        # Parameters
        dataset_id = 12345

        # Mocking
        get_2d_bulk_fake = Mock(
            return_value={
                11111: (np.ones((2, 3)), 20),
                22222: (np.ones((2, 3)), 50),
            }
        )

        get_3d_fake = Mock(
            return_value=np.ones((3, 4)),
        )

        common_queries_fake = Mock()
        common_queries_fake.get_axis.return_value = axis

        get_image_fake = Mock(
            return_value=np.zeros((10, 10)),
        )

        xy_to_pir_fake = Mock(return_value=(1200, 235, 242))

        # Patching
        monkeypatch.setattr("atldld.sync.get_2d_bulk", get_2d_bulk_fake)
        monkeypatch.setattr("atldld.sync.get_3d", get_3d_fake)
        monkeypatch.setattr("atldld.sync.get_image", get_image_fake)
        monkeypatch.setattr("atldld.sync.CommonQueries", common_queries_fake)
        monkeypatch.setattr("atldld.sync.xy_to_pir_API_single", xy_to_pir_fake)

        # Call the function
        gen = download_parallel_dataset(
            dataset_id=dataset_id,
            include_expression=include_expression,
            downsample_ref=downsample_ref,
        )

        # Asserts - preparation
        slice_coordinate_true = 1200 if axis == "coronal" else 242

        if axis == "coronal":
            grid_shape = (8000 / downsample_ref, 11400 / downsample_ref)
        else:
            grid_shape = (13200 / downsample_ref, 8000 / downsample_ref)

        # Asserts - first iteration
        x = next(gen)

        if include_expression:
            assert len(x) == 5

        else:
            assert len(x) == 4

        img_id, slice_coordinate, img, df, *_ = x

        assert img_id == 22222
        assert slice_coordinate == slice_coordinate_true
        assert np.allclose(img, np.zeros((10, 10)))
        assert df.delta_x.shape == grid_shape
        assert df.delta_y.shape == grid_shape

        # Asserts - second iteration
        x = next(gen)

        if include_expression:
            assert len(x) == 5

        else:
            assert len(x) == 4

        img_id, slice_coordinate, img, df, *_ = x

        assert img_id == 11111
        assert slice_coordinate == slice_coordinate_true
        assert np.allclose(img, np.zeros((10, 10)))
        assert df.delta_x.shape == grid_shape
        assert df.delta_y.shape == grid_shape

        # Asserts remaining
        with pytest.raises(StopIteration):
            next(gen)

        assert get_image_fake.call_count == (4 if include_expression else 2)
        assert get_2d_bulk_fake.call_count == 1
        assert get_3d_fake.call_count == 1
        assert common_queries_fake.get_axis.call_count == 1
        assert xy_to_pir_fake.call_count == 2


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
