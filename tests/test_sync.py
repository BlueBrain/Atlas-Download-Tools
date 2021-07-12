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
import requests

from atldld.sync import (
    corners_coronal,
    get_reference_image,
    pir_to_xy_API,
    pir_to_xy_local,
    pir_to_xy_local_with_axis,
    warp_rs9,
    xy_to_pir_API,
)

EXISTING_IMAGE_IDS = [69750516, 101349546]
EXISTING_DATASET_IDS = [479, 1357]  # [Gad, Gfap]


# @pytest.mark.internet
class TestSync:
    """Collections of methods testing the sync module."""

    def test_get_reference_image(self, tmpdir, mocker):
        """Test that it is possible to get reference images"""

        p = 400  # to fit the mocked query

        mocker.patch("os.path.exists", return_value=True)
        mocker.patch(
            "matplotlib.pyplot.imread",
            return_value=np.zeros((8000, 11400), dtype=np.uint8),
        )

        fake_response = Mock(spec=requests.Request)

        rv = {
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

        fake_response.json = Mock(return_value=rv)
        fake_response.ok = True
        mocker.patch("requests.get", return_value=fake_response)

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

    @pytest.mark.internet
    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p", [10, 2000])
    @pytest.mark.parametrize("ds_f", [16, 32])
    def test_warp_rs9(self, p, dataset_id, ds_f):
        """Test that warping works correctly."""

        img_ref_resized, img_section_resized, warped_img_section = warp_rs9(
            p=p, dataset_id=dataset_id, ds_f=ds_f
        )

        expected_shape = (8000 // ds_f, 11400 // ds_f)

        assert img_ref_resized.shape[:2] == expected_shape
        assert img_section_resized.shape[:2] == expected_shape
        assert warped_img_section.shape[:2] == expected_shape

        assert np.all(np.isfinite(img_ref_resized))
        assert np.all(np.isfinite(img_section_resized))
        assert np.all(np.isfinite(warped_img_section))
