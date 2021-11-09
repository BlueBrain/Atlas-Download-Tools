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
import re
from collections import defaultdict

import numpy as np
import pytest
import requests
import responses
from click.testing import CliRunner
from PIL import Image

from atldld.base import DisplacementField
from atldld.cli.download import download_cmd, download_dataset, download_image
from atldld.sync import DatasetNotFoundError


class TestDownloadSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(download_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")


class TestDownloadDataset:
    def test_invalid_dataset_id_errors_out(self, mocker):
        # Mocking
        def fetch_metadata():
            raise DatasetNotFoundError("My Exception")

        mocked_downloader_class = mocker.patch("atldld.sync.DatasetDownloader")
        mocked_downloader_inst = mocked_downloader_class.return_value
        mocked_downloader_inst.fetch_metadata.side_effect = fetch_metadata

        # Testing
        runner = CliRunner()
        result = runner.invoke(download_dataset, ["0", "out"])
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
            download_dataset,
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


class TestDownloadImage:
    @responses.activate
    def test_connection_errors_are_caught(self):
        responses.add(responses.GET, re.compile(r""), requests.ConnectionError())

        # Testing
        runner = CliRunner()
        result = runner.invoke(download_image, ["0", "out"])
        assert result.exit_code != 0  # should exit with an error code
        assert "Error: no network connection" in result.output

    @responses.activate
    def test_nonexistent_image_is_handled(self):
        response = requests.Response()
        response.reason = "some reason"
        response.status_code = 999
        responses.add(
            responses.GET, re.compile(r""), requests.HTTPError(response=response)
        )

        # Testing
        runner = CliRunner()
        result = runner.invoke(download_image, ["0", "out"])
        assert result.exit_code != 0  # should exit with an error code
        assert "Error:" in result.output
        assert response.reason in result.output
        assert str(response.status_code) in result.output

    @responses.activate
    def test_normal_download(self, mocker, tmp_path):
        responses.add(responses.GET, re.compile(r""))

        image_arr = np.random.randint(0, 255, size=(5, 5, 3), dtype=np.uint8)
        image_id = 12345
        output_dir = tmp_path / "out"
        get_image = mocker.patch("atldld.utils.get_image", return_value=image_arr)

        runner = CliRunner()
        result = runner.invoke(download_image, [str(image_id), str(output_dir)])

        assert result.exit_code == 0
        get_image.assert_called_once()

        image_path = output_dir / f"{image_id}-0.png"
        assert image_path.exists()
        with Image.open(image_path) as lazy_img:
            saved_image = np.asarray(lazy_img)
        assert np.array_equal(image_arr, saved_image)

    @responses.activate
    def test_download_with_expression(self, mocker, tmp_path):
        responses.add(responses.GET, re.compile(r""))

        image_arr = np.random.randint(0, 255, size=(5, 5, 3), dtype=np.uint8)
        expression_arr = np.random.randint(0, 255, size=(5, 5, 3), dtype=np.uint8)
        image_id = 12345
        downsample_img = 5

        def get_image(_image_id, folder=None, expression=False, downsample=0):
            if expression:
                return expression_arr
            else:
                return image_arr

        mocker.patch("atldld.utils.get_image", new=get_image)

        runner = CliRunner()
        args = [
            str(image_id),
            str(tmp_path),
            "--include-expression",
            "--downsample-img",
            str(downsample_img),
        ]
        result = runner.invoke(download_image, args)

        assert result.exit_code == 0

        image_path = tmp_path / f"{image_id}-{downsample_img}.png"
        expression_path = tmp_path / f"{image_id}-{downsample_img}-expression.png"
        assert image_path.exists()
        assert expression_path.exists()
        with Image.open(image_path) as lazy_img:
            saved_image = np.asarray(lazy_img)
        with Image.open(expression_path) as lazy_img:
            saved_expression = np.asarray(lazy_img)
        assert np.array_equal(image_arr, saved_image)
        assert np.array_equal(expression_arr, saved_expression)
