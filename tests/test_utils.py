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
"""Test for utils.py module."""
import json
import pathlib
import re
from unittest.mock import Mock

import numpy as np
import pytest
import requests
import responses
from PIL import Image

from atldld.config import user_cache_dir
from atldld.utils import (
    CommonQueries,
    get_2d,
    get_2d_bulk,
    get_3d,
    get_corners_in_ref_space,
    get_experiment_list_from_gene,
    get_image,
    pir_to_xy_API_single,
    xy_to_pir_API_single,
)

EXISTING_IMAGE_IDS = [69750516, 101349546]
EXISTING_DATASET_IDS = [479, 1357]  # [Gad, Gfap]


class TestGetImage:
    @pytest.mark.parametrize("image_id", EXISTING_IMAGE_IDS)  # too slow to try more
    def test_get_image_online(self, image_id, tmpdir, mocker):
        mocker.patch("pathlib.Path.exists", return_value=True)
        fake_img = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))

        # Test getting the normal image
        fake_img.save(pathlib.Path(tmpdir) / f"{image_id}-0.jpg")
        img = get_image(image_id, tmpdir)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8

        # Test getting the expression image
        fake_img.save(pathlib.Path(tmpdir) / f"{image_id}-0-expression.jpg")
        img = get_image(image_id, tmpdir, expression=True)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8

    def test_get_image_offline(self, tmpdir, img):
        """Test whether offline loading works"""
        image_id_fake = 123456

        img_path = pathlib.Path(tmpdir) / f"{image_id_fake}-0.jpg"
        Image.fromarray(img, mode="L").save(img_path)

        img = get_image(image_id_fake, str(tmpdir))
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8

    def test_decompression_bomb_warning_suppressed(self, tmpdir):
        image_id = 123

        # Prepare a cached image. The decompression bomb warning is triggered
        # at about 90M pixels, so we prepare an image with 100M pixels.
        img = Image.fromarray(np.zeros((10_000, 10_000), dtype=np.uint8))
        img.save(pathlib.Path(tmpdir) / f"{image_id}-0.jpg")

        with pytest.warns(None) as records:
            get_image(image_id, folder=str(tmpdir))

        assert len(records) == 0

    def test_decompression_bomb_error_triggered(self, tmpdir):
        image_id = 123

        # Prepare a cached image. The decompression bomb error is triggered
        # at about 180M pixels, so we prepare an image with 200M pixels.
        img = Image.fromarray(np.zeros((20_000, 10_000), dtype=np.uint8))
        img.save(pathlib.Path(tmpdir) / f"{image_id}-0.jpg")

        with pytest.raises(Image.DecompressionBombError):
            get_image(image_id, folder=str(tmpdir))


def test_get_corners_in_ref_space(mocker):
    image_id = 123
    width = 200
    height = 100

    def xy_to_pir(x, y, _image_id):
        return x * 10, y * 10, 53

    mocker.patch("atldld.utils.xy_to_pir_API_single", xy_to_pir)

    corners = get_corners_in_ref_space(image_id, width, height)
    assert isinstance(corners, np.ndarray)
    assert np.all(
        corners
        == [
            (0, 0, 53),
            (width * 10, 0, 53),
            (width * 10, height * 10, 53),
            (0, height * 10, 53),
        ]
    )


class TestXyToPirApiSingle:
    @pytest.mark.internet
    @pytest.mark.parametrize("image_id", EXISTING_IMAGE_IDS)
    def test_fetching_online_no_mocking(self, image_id):
        x, y = np.random.randint(0, 10000, 2)
        p, i, r = xy_to_pir_API_single(x, y, image_id=image_id)
        assert np.isfinite(p)
        assert np.isfinite(i)
        assert np.isfinite(r)

    @responses.activate
    def test_fetching_online_and_storing_in_cache(self):
        # Mock the online response
        image_id = 123
        x, y = 10, 40
        pir = (-983.5501122512056, 335.1538167934086, 4117.19706627407)
        rv = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": {"image_to_reference": {"x": pir[0], "y": pir[1], "z": pir[2]}},
        }
        responses.add(responses.GET, re.compile(r".*/image_to_reference/.*"), json=rv)

        # Check the returned value
        pir_got = xy_to_pir_API_single(x, y, image_id=image_id)
        assert pir_got == pir

        # Check that the cache is updated
        cache_file = user_cache_dir() / "image-to-reference" / f"{image_id}.json"
        assert cache_file.exists()
        with cache_file.open() as fp:
            cache_json = json.load(fp)
        assert cache_json.get(f"x={x}&y={y}") == list(pir)

    @responses.activate
    def test_reading_from_cache(self):
        # Because we activated responses but didn't add any response values any
        # request will raise an error. This way we automatically test that no
        # requests are sent and the values are obtained offline from the cache
        image_id = 123
        x, y = 5, 10
        pir = 20.5, 13.7, 8.8

        # Prepare the cache file
        cache_file = user_cache_dir() / "image-to-reference" / f"{image_id}.json"
        cache_file.parent.mkdir()
        cache_json = {f"x={x}&y={y}": pir}
        with cache_file.open("w") as fp:
            json.dump(cache_json, fp)

        # Run the test
        pir_got = xy_to_pir_API_single(x, y, image_id=image_id)
        assert pir_got == pir


class TestUtils:
    @pytest.mark.internet
    def test_get_experiment_list(self):
        """Test that experiment list is good."""
        results = get_experiment_list_from_gene("S100b", "sagittal")
        assert [79762548, 922, 924] == results
        results = get_experiment_list_from_gene("S100b", "coronal")
        assert [923, 79591593] == results

    @pytest.mark.parametrize(
        "image_id", EXISTING_IMAGE_IDS[:1]
    )  # doesnt make sense to do more because we mock
    @pytest.mark.parametrize(
        "add_last", [True, False], ids=["add_last", "dont_add_last"]
    )
    @pytest.mark.parametrize("ref2inp", [True, False], ids=["ref2inp", "inp2ref"])
    def test_get_2d(self, image_id, ref2inp, add_last, mocker):
        """Test that it is possible to get 2d transformations."""

        # patching
        fake_response = Mock(spec=requests.Request)
        fake_response.ok = True

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
                    "bits_per_component": 8,
                    "data_set_id": 69782969,
                    "expression_path": "whatever",
                    "failed": False,
                    "height": 7297,
                    "id": 69750516,
                    "image_height": 7297,
                    "image_type": "Primary",
                    "image_width": 14065,
                    "isi_experiment_id": None,
                    "lims1_id": 69750516,
                    "number_of_components": 3,
                    "ophys_experiment_id": None,
                    "path": "whatever",  #
                    "projection_function": None,
                    "resolution": 1.049,
                    "section_number": 78,
                    "specimen_id": None,
                    "structure_id": None,
                    "tier_count": 7,
                    "width": 14065,
                    "x": 0,
                    "y": 0,
                    "alignment2d": {
                        "id": 345778741,
                        "section_image_id": 69750516,
                        "tsv_00": 1.04898,
                        "tsv_01": -0.00572724,
                        "tsv_02": 0.00572724,
                        "tsv_03": 1.04898,
                        "tsv_04": 505.522,
                        "tsv_05": 2024.92,
                        "tvs_00": 0.953275,
                        "tvs_01": 0.00520468,
                        "tvs_02": -0.00520468,
                        "tvs_03": 0.953275,
                        "tvs_04": -492.44,
                        "tvs_05": -1927.67,
                    },
                }
            ],
        }

        fake_response.json = Mock(return_value=rv)
        mocker.patch("requests.get", return_value=fake_response)

        # Actual function
        a = get_2d(image_id, ref2inp=ref2inp, add_last=add_last)

        # Mock calls
        assert (
            fake_response.json.call_count == 1
        )  # assert_called not available in python 3.5 :(

        assert isinstance(a, np.ndarray)
        assert a.shape == ((3, 3) if add_last else (2, 3))
        assert np.all(np.isfinite(a))

    @pytest.mark.parametrize(
        "dataset_id", EXISTING_DATASET_IDS[:1]
    )  # doesnt make sense to do more because we mock
    @pytest.mark.parametrize(
        "add_last", [True, False], ids=["add_last", "dont_add_last"]
    )
    @pytest.mark.parametrize("ref2inp", [True, False], ids=["ref2inp", "inp2ref"])
    def test_get_2d_bulk(self, dataset_id, ref2inp, add_last, mocker):
        """Test that it is possible to get 2d transformations for the entire dataset"""

        # patching
        fake_response = Mock(spec=requests.Request)
        fake_response.ok = True

        rv = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": 2,
            "total_rows": 56,
            "msg": [
                {
                    "annotated": False,
                    "axes": None,
                    "bits_per_component": 8,
                    "data_set_id": 479,
                    "expression_path": "/external/aibssan/production32/prod38/"
                    "b.479.1/b.479.1_4_expression.aff",
                    "failed": False,
                    "height": 3864,
                    "id": 101349756,
                    "image_height": 3864,
                    "image_type": "Primary",
                    "image_width": 4232,
                    "isi_experiment_id": None,
                    "lims1_id": 89792,
                    "number_of_components": 3,
                    "ophys_experiment_id": None,
                    "path": "/external/aibssan/production4/Gad1_Baylor_9231/"
                    "zoomify/primary/None/Gad1_4_None_A.aff",
                    "projection_function": None,
                    "resolution": 1.63,
                    "section_number": 4,
                    "specimen_id": None,
                    "structure_id": None,
                    "tier_count": 6,
                    "width": 4232,
                    "x": 0,
                    "y": 0,
                    "alignment2d": {
                        "id": 348273717,
                        "section_image_id": 101349756,
                        "tsv_00": 1.62982,
                        "tsv_01": 0.0243755,
                        "tsv_02": -0.0243755,
                        "tsv_03": 1.62982,
                        "tsv_04": 3051.75,
                        "tsv_05": 2776.39,
                        "tvs_00": 0.613428,
                        "tvs_01": -0.0091744,
                        "tvs_02": 0.0091744,
                        "tvs_03": 0.613428,
                        "tvs_04": -1846.56,
                        "tvs_05": -1731.11,
                    },
                },
                {
                    "annotated": False,
                    "axes": None,
                    "bits_per_component": 8,
                    "data_set_id": 479,
                    "expression_path": "/external/aibssan/production32/prod38/"
                    "b.479.1/b.479.1_12_expression.aff",
                    "failed": False,
                    "height": 4016,
                    "id": 101349764,
                    "image_height": 4016,
                    "image_type": "Primary",
                    "image_width": 4536,
                    "isi_experiment_id": None,
                    "lims1_id": 84014,
                    "number_of_components": 3,
                    "ophys_experiment_id": None,
                    "path": "/external/aibssan/production4/Gad1_Baylor_9231/"
                    "zoomify/primary/None/Gad1_12_None_B.aff",
                    "projection_function": None,
                    "resolution": 1.63,
                    "section_number": 12,
                    "specimen_id": None,
                    "structure_id": None,
                    "tier_count": 6,
                    "width": 4536,
                    "x": 0,
                    "y": 0,
                    "alignment2d": {
                        "id": 348273722,
                        "section_image_id": 101349764,
                        "tsv_00": 1.62992,
                        "tsv_01": 0.0163663,
                        "tsv_02": -0.0163663,
                        "tsv_03": 1.62992,
                        "tsv_04": 2800.76,
                        "tsv_05": 2608.26,
                        "tvs_00": 0.613466,
                        "tvs_01": -0.00615993,
                        "tvs_02": 0.00615993,
                        "tvs_03": 0.613466,
                        "tvs_04": -1702.11,
                        "tvs_05": -1617.33,
                    },
                },
            ],
        }

        fake_response.json = Mock(return_value=rv)
        mocker.patch("requests.get", return_value=fake_response)

        res = get_2d_bulk(dataset_id, ref2inp=ref2inp, add_last=add_last)

        # Mock calls
        assert (
            fake_response.json.call_count == 1
        )  # assert_called not available in python 3.5 :(

        sn_list = [sn for _, sn in res.values()]
        image_id_list = list(res.keys())

        for _, (a, section_number) in res.items():
            assert isinstance(a, np.ndarray)
            assert a.shape == ((3, 3) if add_last else (2, 3))
            assert np.all(np.isfinite(a))
            assert np.isfinite(section_number)

        assert len(sn_list) == len(set(sn_list))  # Make sure no duplicates
        assert len(image_id_list) == len(set(image_id_list))  # Make sure no duplicates

    @pytest.mark.parametrize(
        "dataset_id", EXISTING_DATASET_IDS[:1]
    )  # doesnt make sense to do more because we mock
    @pytest.mark.parametrize(
        "add_last", [True, False], ids=["add_last", "dont_add_last"]
    )
    @pytest.mark.parametrize(
        "return_meta", [True, False], ids=["return_meta", "dont_return_meta"]
    )
    @pytest.mark.parametrize("ref2inp", [True, False], ids=["ref2inp", "inp2ref"])
    def test_get_3d(self, dataset_id, ref2inp, add_last, return_meta, mocker):
        """Test that it is possible to get 3d transformations."""

        # patching
        fake_response = Mock(spec=requests.Request)
        fake_response.ok = True

        rv = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": [
                {
                    "blue_channel": None,
                    "delegate": False,
                    "expression": False,
                    "failed": False,
                    "failed_facet": 734881840,
                    "green_channel": None,
                    "id": 479,
                    "name": None,
                    "plane_of_section_id": 1,
                    "qc_date": None,
                    "red_channel": None,
                    "reference_space_id": 9,
                    "rnaseq_design_id": None,
                    "section_thickness": 25.0,
                    "specimen_id": 702765,
                    "sphinx_id": 138110,
                    "storage_directory": "/external/mouse/prod2/image_series_479/",
                    "weight": 5470,
                    "alignment3d": {
                        "aligned_id": 479,
                        "aligned_type": "DataSet",
                        "id": 348273183,
                        "trv_00": 0.0,
                        "trv_01": -0.0109026,
                        "trv_02": 0.930036,
                        "trv_03": 0.0,
                        "trv_04": 1.02989,
                        "trv_05": 0.00901046,
                        "trv_06": -1.00888,
                        "trv_07": 0.0,
                        "trv_08": 0.0,
                        "trv_09": 1257.98,
                        "trv_10": 1586.18,
                        "trv_11": 12854.5,
                        "tvr_00": 0.0,
                        "tvr_01": 0.0,
                        "tvr_02": -0.991197,
                        "tvr_03": -0.00940617,
                        "tvr_04": 0.97088,
                        "tvr_05": 0.0,
                        "tvr_06": 1.07512,
                        "tvr_07": 0.0113814,
                        "tvr_08": 0.0,
                        "tvr_09": 12741.3,
                        "tvr_10": -1528.16,
                        "tvr_11": -1370.53,
                    },
                }
            ],
        }

        fake_response.json = Mock(return_value=rv)
        mocker.patch("requests.get", return_value=fake_response)

        if not return_meta:
            a = get_3d(
                dataset_id, ref2inp=ref2inp, add_last=add_last, return_meta=return_meta
            )
        else:
            a, rs, st = get_3d(
                dataset_id, ref2inp=ref2inp, add_last=add_last, return_meta=return_meta
            )

        # Assertions

        assert isinstance(a, np.ndarray)
        assert a.shape == ((4, 4) if add_last else (3, 4))
        assert np.all(np.isfinite(a))

        if return_meta:
            assert np.isfinite(rs)
            assert np.isfinite(st)

    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS[:1])
    @pytest.mark.parametrize("p", [10])
    @pytest.mark.parametrize("i", [40])
    @pytest.mark.parametrize("r", [100])
    def test_pir_to_xy_API_single(self, dataset_id, p, i, r, mocker):
        """Test that pir to xy API works."""

        # patching
        fake_response = Mock(spec=requests.Request)
        fake_response.ok = True

        rv = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": 0,
            "total_rows": 0,
            "msg": [
                {
                    "image_sync": {
                        "section_data_set_id": "479",
                        "section_image_id": 101350196,
                        "section_number": 444,
                        "x": -1750.838222501794,
                        "y": -1204.1704347987138,
                    }
                }
            ],
        }

        fake_response.json = Mock(return_value=rv)
        mocker.patch("requests.get", return_value=fake_response)

        x, y, section_number, closest_section_image_id = pir_to_xy_API_single(
            p, i, r, dataset_id=dataset_id
        )

        # Mock calls
        assert (
            fake_response.json.call_count == 1
        )  # assert_called not available in python 3.5 :(

        assert np.isfinite(x)
        assert np.isfinite(y)
        assert np.isfinite(section_number)
        assert np.isfinite(closest_section_image_id)
        assert isinstance(closest_section_image_id, int)

    @pytest.mark.internet
    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    @pytest.mark.parametrize("p", [100, 100.1])
    @pytest.mark.parametrize("i", [400, 1213.1])
    @pytest.mark.parametrize("r", [4900, 7223.1])
    def test_pir_to_xy_APIequalsALLENSDK(self, dataset_id, p, i, r):
        """Test that extracted requests equals official Allen's package result."""

        x, y, section_number, closest_section_image_id = pir_to_xy_API_single(
            p, i, r, dataset_id=dataset_id
        )

        url = (
            "https://api.brain-map.org/api/v2/reference_to_image/9.json"
            f"?x={p}&y={i}&z={r}&section_data_set_ids={dataset_id}"
        )

        response = requests.get(url)
        r = response.json()["msg"][0]["image_sync"]

        x_s, y_s, section_number_s, closest_section_image_id_s = (
            r["x"],
            r["y"],
            r["section_number"],
            r["section_image_id"],
        )

        assert x == x_s
        assert y == y_s
        assert section_number == section_number_s
        assert closest_section_image_id == closest_section_image_id_s


# @pytest.mark.internet
class TestCommonQueries:
    """A set of tests focused on some basic queries in the CommonQueries class."""

    @pytest.mark.parametrize("dataset_id", EXISTING_DATASET_IDS)
    def test_get_reference_space(self, dataset_id):
        """Test that all testing datasets are from reference space 9."""

        assert CommonQueries.get_reference_space(dataset_id) == 9
