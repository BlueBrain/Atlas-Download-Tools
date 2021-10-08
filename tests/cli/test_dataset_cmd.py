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
import json
import pathlib
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict

import click
import numpy as np
import pytest
import responses
from click.testing import CliRunner

from atldld.base import DisplacementField
from atldld.cli.dataset import (
    dataset_cmd,
    dataset_download,
    dataset_info,
    dataset_preview,
    get_dataset_meta_or_abort,
)
from atldld.sync import DatasetNotFoundError


class TestDatasetSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(dataset_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")


class TestGetDatasetMetaOrAbortHelper:
    @responses.activate
    def test_it_works_like_intended(self):
        meta = {"info": "This would be the dataset metadata"}
        response_json = {
            "id": 0,
            "success": True,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": [meta],
        }
        responses.add(responses.GET, re.compile(""), json=response_json)
        meta_got = get_dataset_meta_or_abort(0)
        assert meta_got == meta

    @responses.activate
    def test_rma_errors_are_caught(self):
        # this should lead to an RMAError in atldld.requests.rma_all
        response_json = {"success": False, "msg": "Some error"}
        responses.add(responses.GET, re.compile(""), json=response_json)
        with pytest.raises(click.ClickException, match=r"An error occurred"):
            get_dataset_meta_or_abort(0)

    def test_invalid_dataset_id_aborts(self, mocker):
        # An empty msg=[] means the dataset ID was not found
        mocker.patch("atldld.requests.rma_all", return_value=[])
        with pytest.raises(click.ClickException, match=r"does not exist"):
            get_dataset_meta_or_abort(0)

    def test_multiple_datasets_returned_aborts(self, mocker):
        # Return two datasets for one query - something is wrong.
        msg = ["dataset1", "dataset2"]
        mocker.patch("atldld.requests.rma_all", return_value=msg)
        with pytest.raises(click.ClickException, match=r"more than one dataset"):
            get_dataset_meta_or_abort(0)

    def test_metadata_not_a_dict_aborts(self, mocker):
        msg = ["This should be a dict"]
        mocker.patch("atldld.requests.rma_all", return_value=msg)
        with pytest.raises(
            click.ClickException, match=r"unexpected dataset information format"
        ):
            get_dataset_meta_or_abort(0)

    def test_metadata_keys_not_strings_aborts(self, mocker):
        msg = [{1: "The keys should all be strings!"}]
        mocker.patch("atldld.requests.rma_all", return_value=msg)
        with pytest.raises(
            click.ClickException, match=r"unexpected dataset information format"
        ):
            get_dataset_meta_or_abort(0)


class TestDatasetInfo:
    def test_calling_without_parameters_prints_usage(self):
        runner = CliRunner()
        result = runner.invoke(dataset_info)
        assert result.exit_code != 0  # should exit with an error code
        assert result.output.startswith("Usage:")

    @responses.activate
    def test_normal_usage_works_as_intended(self):
        dataset_info_json: Dict[str, Any] = {
            "blue_channel": None,
            "expression": False,
            "failed": False,
            "green_channel": None,
            "id": 123,
            "name": None,
            "plane_of_section_id": 1,
            "red_channel": None,
            "reference_space_id": 9,
            "section_thickness": 25.0,
            "specimen_id": 456,
            "sphinx_id": 789,
            "genes": [{"acronym": "Gad1"}],
            "section_images": [{}, {}, {}],
        }
        response_json = {
            "id": 0,
            "success": True,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": [dataset_info_json],
        }
        expected_output = """
        ID                       : 123
        Sphinx ID                : 789
        Specimen ID              : 456
        Name                     : -
        Failed                   : No
        Expression               : No
        Gene(s)                  : Gad1
        RGB channels             : - / - / -
        Section thickness        : 25.0µm
        Plane of section         : coronal
        Number of section images : 3
        Reference space          : 9 (P56 Brain)
        """
        responses.add(responses.GET, re.compile(""), json=response_json)
        runner = CliRunner()
        result = runner.invoke(dataset_info, [str(dataset_info_json["id"])])
        assert result.exit_code == 0
        assert result.output.strip() == textwrap.dedent(expected_output).strip()


class TestDatasetPreview:
    @pytest.fixture
    def dataset_meta_response(self):
        dataset_info_json: Dict[str, Any] = {
            "plane_of_section_id": 1,
            "section_images": [
                {"id": 1, "section_number": 1, "image_width": 200, "image_height": 100},
                {"id": 2, "section_number": 2, "image_width": 200, "image_height": 100},
                {"id": 3, "section_number": 3, "image_width": 200, "image_height": 100},
            ],
        }
        response_json = {
            "id": 0,
            "success": True,
            "start_row": 0,
            "num_rows": 1,
            "total_rows": 1,
            "msg": [dataset_info_json],
        }

        return response_json

    def test_calling_without_parameters_prints_usage(self):
        runner = CliRunner()
        result = runner.invoke(dataset_preview)
        assert result.exit_code != 0  # should exit with an error code
        assert result.output.startswith("Usage:")

    @responses.activate
    def test_normal_usage_works_as_intended(self, mocker, dataset_meta_response):
        mocked_get_corners_in_ref_space = mocker.patch(
            "atldld.utils.get_corners_in_ref_space"
        )
        mocked_dataset_preview = mocker.patch("atldld.plot.dataset_preview")
        responses.add(responses.GET, re.compile(""), json=dataset_meta_response)

        runner = CliRunner()
        result = runner.invoke(dataset_preview, ["123"])
        assert result.exit_code == 0
        assert mocked_get_corners_in_ref_space.call_count == 3
        assert mocked_dataset_preview.call_count == 1
        fig = mocked_dataset_preview.return_value
        fig.savefig.assert_called_once()

    @responses.activate
    def test_custom_output_directory_works(self, mocker, dataset_meta_response, tmpdir):
        mocked_get_corners_in_ref_space = mocker.patch(
            "atldld.utils.get_corners_in_ref_space"
        )
        mocked_dataset_preview = mocker.patch("atldld.plot.dataset_preview")
        responses.add(responses.GET, re.compile(""), json=dataset_meta_response)

        # Create a subdirectory for the output to check if it will be created
        output_dir = pathlib.Path(tmpdir) / "new-subdir"
        assert not output_dir.exists()

        runner = CliRunner()
        result = runner.invoke(
            dataset_preview, ["123", "--output-dir", str(output_dir)]
        )

        assert result.exit_code == 0
        assert mocked_get_corners_in_ref_space.call_count == 3
        assert mocked_dataset_preview.call_count == 1
        fig = mocked_dataset_preview.return_value
        fig.savefig.assert_called_with(
            pathlib.Path(output_dir) / "dataset-id-123-preview.png"
        )
        assert output_dir.exists()


class TestDatasetDownload:
    def test_invalid_dataset_id_errors_out(self, mocker):
        # Mocking
        def fetch_metadata():
            raise DatasetNotFoundError("My Exception")

        mocked_downloader_class = mocker.patch("atldld.sync.DatasetDownloader")
        mocked_downloader_inst = mocked_downloader_class.return_value
        mocked_downloader_inst.fetch_metadata.side_effect = fetch_metadata

        # Testing
        runner = CliRunner()
        result = runner.invoke(dataset_download, ["0", "out"])
        assert result.exit_code != 0  # should exit with an error code
        assert "Error: My Exception" in result.output

    @pytest.mark.parametrize("dataset_id", [3532, 542133])
    @pytest.mark.parametrize("n_images", [2, 3])
    @pytest.mark.parametrize("include_expression", [True, False])
    @pytest.mark.parametrize("folder_exists", [True, False])
    def test_all(
        self,
        tmp_path,
        mocker,
        dataset_id,
        n_images,
        include_expression,
        folder_exists,
    ):
        dataset_id = str(dataset_id)
        runner = CliRunner()
        output_folder = tmp_path / "output_folder"
        if folder_exists:
            output_folder.mkdir()

        # Mocking and patching
        mocked_downloader_class = mocker.patch("atldld.sync.DatasetDownloader")
        mocked_downloader_inst = mocked_downloader_class.return_value
        mocked_downloader_inst.__len__.return_value = n_images
        mocked_downloader_inst.metadata = defaultdict(lambda: defaultdict(lambda: 1))

        def fake_run():
            for i in range(n_images):
                image_id = i
                section_coordinate = i * 2.5
                img = np.ones((100, 100, 3))
                img_expr = np.ones((100, 100, 3)) if include_expression else None
                df = DisplacementField(np.zeros((100, 100)), np.zeros((100, 100)))

                yield image_id, section_coordinate, img, img_expr, df

        mocked_downloader_inst.run.side_effect = fake_run

        # Invoking the CLI
        result = runner.invoke(
            dataset_download,
            [
                dataset_id,
                str(output_folder),
            ],
            catch_exceptions=False,
        )

        # Asserts
        assert result.exit_code == 0
        assert dataset_id in result.output

        mocked_downloader_class.assert_called()
        mocked_downloader_inst.fetch_metadata.assert_called()
        mocked_downloader_inst.run.assert_called()

        img_paths = [p for p in output_folder.iterdir() if p.suffix == ".png"]
        assert len(img_paths) == (int(include_expression) + 1) * n_images

        metadata_path = output_folder / "metadata.json"
        assert metadata_path.exists()

        with metadata_path.open() as f:
            metadata = json.load(f)

        assert {
            "dataset_id",
            "downsample_ref",
            "downsample_img",
            "downsample_img",
            "plane_of_section",
            "section_thickness",
            "per_image",
        } == set(metadata.keys())

        assert len(metadata["per_image"]) == n_images
        for image_metadata in metadata["per_image"].values():
            assert {
                "section_coordinate",
                "section_coordinate_scaled",
            } == set(image_metadata.keys())
