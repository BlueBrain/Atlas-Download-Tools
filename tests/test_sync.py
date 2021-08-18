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
import re
from unittest.mock import Mock

import numpy as np
import pytest
import responses

from atldld.sync import (
    corners_coronal,
    download_dataset_parallel,
    get_reference_image,
    get_transform_parallel,
    pir_to_xy_API,
    pir_to_xy_local,
    pir_to_xy_local_with_axis,
    xy_to_pir_API,
)

EXISTING_IMAGE_IDS = [69750516, 101349546]
EXISTING_DATASET_IDS = [479, 1357]  # [Gad, Gfap]


# @pytest.mark.internet
class TestSync:
    """Collections of methods testing the sync module."""

    @responses.activate
    def test_get_reference_image(self, tmpdir, mocker):
        """Test that it is possible to get reference images"""

        p = 400  # to fit the mocked query

        mocker.patch("os.path.exists", return_value=True)
        mocker.patch(
            "matplotlib.pyplot.imread",
            return_value=np.zeros((8000, 11400), dtype=np.uint8),
        )

        response_json = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": [
                {
                    "annotated": False,
                    "axes": None,
                    "bits_per_component": None,
                    "data_set_id": 576985993,
                    "expression_path": None,
                    "failed": False,
                    "height": 8000,
                    "id": 576989001,
                    "image_height": 8000,
                    "image_type": "Primary",
                    "image_width": 11400,
                    "isi_experiment_id": None,
                    "lims1_id": None,
                    "number_of_components": None,
                    "ophys_experiment_id": None,
                    "path": "/external/ctyconn/prod38"
                    "/9900600001-0401_576928557/9900600001-0401.aff",
                    "projection_function": None,
                    "resolution": 1.0,
                    "section_number": 400,
                    "specimen_id": None,
                    "structure_id": None,
                    "tier_count": 7,
                    "width": 11400,
                    "x": 0,
                    "y": 0,
                }
            ],
        }
        responses.add(responses.GET, re.compile(""), json=response_json)

        if p % 10 == 0 and 0 <= p < 13200:
            img = get_reference_image(p, tmpdir)

            assert isinstance(img, np.ndarray)
            assert img.shape[:2] == (8000, 11400)
            assert np.all(np.isfinite(img))

        else:
            with pytest.raises(ValueError):
                get_reference_image(p, tmpdir)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p_list", [[10, 132.3]])
    @pytest.mark.parametrize("i_list", [[40, 31.8]])
    @pytest.mark.parametrize("r_list", [[49, 12.3]])
    def test_pir_to_xy_API(self, p_list, i_list, r_list, dataset_id):
        """Test that it works."""

        p_list_wrong = [p_list[0]] * (len(p_list) + 1)

        with pytest.raises(ValueError):
            pir_to_xy_API(p_list_wrong, i_list, r_list, dataset_id=dataset_id)

        (
            x_list,
            y_list,
            section_number_list,
            closest_section_image_id_list,
        ) = pir_to_xy_API(p_list, i_list, r_list, dataset_id=dataset_id)

        lng = len(p_list)
        assert (
            len(x_list)
            == len(y_list)
            == len(section_number_list)
            == len(closest_section_image_id_list)
            == lng
        )

        for x, y, section_number, closest_section_image_id in zip(
            x_list, y_list, section_number_list, closest_section_image_id_list
        ):
            assert np.isfinite(x)
            assert np.isfinite(y)
            assert np.isfinite(section_number)
            assert np.isfinite(closest_section_image_id)
            assert isinstance(closest_section_image_id, int)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p_list", [[10, 132.3]])
    @pytest.mark.parametrize("i_list", [[40, 31.8]])
    @pytest.mark.parametrize("r_list", [[49, 12.3]])
    def test_pir_to_xy_local(self, p_list, i_list, r_list, dataset_id):
        """Test that local works."""

        p_list_wrong = [p_list[0]] * (len(p_list) + 1)

        with pytest.raises(ValueError):
            pir_to_xy_API(p_list_wrong, i_list, r_list, dataset_id=dataset_id)

        (
            x_list,
            y_list,
            section_number_list,
            closest_section_image_id_list,
        ) = pir_to_xy_local(p_list, i_list, r_list, dataset_id=dataset_id)

        lng = len(p_list)
        assert (
            len(x_list)
            == len(y_list)
            == len(section_number_list)
            == len(closest_section_image_id_list)
            == lng
        )

        for x, y, section_number, closest_section_image_id in zip(
            x_list, y_list, section_number_list, closest_section_image_id_list
        ):
            assert np.isfinite(x)
            assert np.isfinite(y)
            assert np.isfinite(section_number)
            assert np.isfinite(closest_section_image_id)
            assert isinstance(closest_section_image_id, int)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p", [10, 20])
    @pytest.mark.parametrize("i_list", [[40, 31.8]])
    @pytest.mark.parametrize("r_list", [[49, 12.3]])
    def test_pir_to_xy_local_with_axis(self, p, i_list, r_list, dataset_id):
        """Test that local works."""

        N = len(i_list)
        p_list = [p] * N

        (
            x_list,
            y_list,
            section_number,
            closest_section_image_id,
        ) = pir_to_xy_local_with_axis(p_list, i_list, r_list, dataset_id=dataset_id)

        assert len(x_list) == len(y_list) == len(i_list) == len(r_list)

        assert isinstance(section_number, int)
        assert np.isfinite(section_number)
        assert isinstance(closest_section_image_id, int)
        assert np.isfinite(closest_section_image_id)

        for x, y in zip(x_list, y_list):
            assert np.isfinite(x)
            assert np.isfinite(y)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p_list", [[10, 132.3, 1394]])
    @pytest.mark.parametrize("i_list", [[40, 31.8, 1230]])
    @pytest.mark.parametrize("r_list", [[49, 12.3, 3001.1]])
    def test_pir_to_xy_localequalsAPI(self, p_list, i_list, r_list, dataset_id):
        """Test that local is the same as API."""

        (
            x_list_l,
            y_list_l,
            section_number_list_l,
            closest_section_image_id_list_l,
        ) = pir_to_xy_local(
            p_list, i_list, r_list, dataset_id=dataset_id
        )  # noqa

        (
            x_list_a,
            y_list_a,
            section_number_list_a,
            closest_section_image_id_list_a,
        ) = pir_to_xy_API(
            p_list, i_list, r_list, dataset_id=dataset_id
        )  # noqa

        assert x_list_l == x_list_a
        assert y_list_l == y_list_a
        assert section_number_list_l == section_number_list_a
        assert closest_section_image_id_list_l == closest_section_image_id_list_a

    @pytest.mark.internet
    @pytest.mark.parametrize("image_id", EXISTING_IMAGE_IDS)
    @pytest.mark.parametrize("x_list", [[10, 132.3]])
    @pytest.mark.parametrize("y_list", [[40, 31.8]])
    def test_xy_to_pir_API(self, x_list, y_list, image_id):
        """Test that it works."""

        x_list_wrong = [x_list[0]] * (len(x_list) + 1)

        with pytest.raises(ValueError):
            xy_to_pir_API(x_list_wrong, y_list, image_id=image_id)

        p_list, i_list, r_list = xy_to_pir_API(x_list, y_list, image_id=image_id)

        assert len(p_list) == len(i_list) == len(r_list) == len(x_list) == len(y_list)

        for p, i, r in zip(p_list, i_list, r_list):
            assert np.isfinite(p)
            assert np.isfinite(i)
            assert np.isfinite(r)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p", [10, 2000])
    def test_corners_coronal(self, p, dataset_id):
        """Test that corners in rs 9 work"""

        section_number, closest_section_image_id = corners_coronal(p, dataset_id)

        assert isinstance(section_number, int)
        assert isinstance(closest_section_image_id, int)
        assert np.isfinite(section_number)
        assert np.isfinite(closest_section_image_id)


class TestGetTransformParallel:
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

        df = get_transform_parallel(
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

        assert x_pred == pytest.approx(x, abs=1e-3)
        assert y_pred == pytest.approx(y, abs=1e-3)


class TestDownloadDatasetParallel:
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
        gen = download_dataset_parallel(
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
