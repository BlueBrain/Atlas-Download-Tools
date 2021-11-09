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
import re

import pytest
from click.testing import CliRunner

from atldld.cli.search import search_cmd, search_dataset, search_image
from atldld.requests import RMAError


class TestSearchSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(search_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")


@pytest.fixture
def rma_all(mocker):
    return mocker.patch("atldld.requests.rma_all", return_value=[])


class TestSearchDataset:
    def test_calling_without_parameters_produces_an_error(self):
        runner = CliRunner()
        result = runner.invoke(search_dataset)
        assert result.exit_code != 0  # should exit with an error code
        assert result.output.startswith(
            "Error: At least one of the search criteria has to be specified."
        )

    def test_rma_errors_are_reported(self, rma_all):
        error_msg = "Some error occurred"
        rma_all.side_effect = RMAError(error_msg)
        result = CliRunner().invoke(search_dataset, ["--id", "1"])
        assert result.exit_code != 0
        assert "error" in result.output
        assert error_msg in result.output

    def test_no_datasets_found(self, rma_all):
        rma_all.return_value = []
        result = CliRunner().invoke(search_dataset, ["--id", "1"])
        assert result.exit_code == 0
        assert "No datasets found" in result.output

    def test_all_results_are_shown(self, rma_all):
        msg = [
            {
                "id": 1,
                "plane_of_section_id": 1,
                "genes": [{"acronym": "Gad1"}],
                "section_images": [1, 2, 3],
            },
            {
                "id": 2,
                "plane_of_section_id": 1,
                "genes": [{"acronym": "Memo1"}],
                "section_images": [1],
            },
            {
                "id": 3,
                "plane_of_section_id": 1,
                "genes": [{"acronym": "Pvalb"}],
                "section_images": [1, 2, 3, 4, 5],
            },
        ]
        rma_all.return_value = msg
        runner = CliRunner()
        result = runner.invoke(search_dataset, ["--id", "whatever"])
        assert result.exit_code == 0

        # Check the output contains the correct number of bullet points
        assert len(re.findall(r"\*", result.output)) == len(msg)

        # Check each bullet point has the correct content
        for item in msg:
            genes = item["genes"][0]["acronym"]  # type: ignore[index]
            n_img = len(item["section_images"])  # type: ignore[arg-type]
            assert re.search(
                (
                    fr"id: +{item['id']}, genes: +{genes}, "
                    fr"coronal, {n_img} section images"
                ),
                result.output.strip(),
            )

    @pytest.mark.parametrize(
        ("cli_params", "expected_criteria"),
        (
            (["--id", "1"], {"id": "1"}),
            (["--specimen", "789"], {"specimen_id": "789"}),
            (["--gene-name", "my-gene"], {"genes": {"acronym": "my-gene"}}),
            (
                ["--plane-of-section", "coronal"],
                {"plane_of_section": {"name": "coronal"}},
            ),
        ),
        ids=(
            "Filter by dataset ID",
            "Filter by specimen ID",
            "Filter by gene acronym",
            "Filter by plane of section",
        ),
    )
    def test_search_filters(self, rma_all, cli_params, expected_criteria):
        """Test that CLI parameters are correctly translated to criteria."""
        result = CliRunner().invoke(search_dataset, cli_params)
        assert result.exit_code == 0
        assert rma_all.called_once
        # Get the args of the last call to rma_all
        (rma_parameters,), _kwargs = rma_all.call_args
        assert rma_parameters.criteria == expected_criteria

    def test_unknown_plane_of_section(self, rma_all):
        """Test that CLI parameters are correctly translated to criteria."""
        plane_of_section = "unknown value"
        result = CliRunner().invoke(
            search_dataset,
            ["--plane-of-section", plane_of_section],
        )
        assert result.exit_code == 0
        assert f'Unknown plane of section name: "{plane_of_section}"' in result.output


class TestSearchImage:
    def test_calling_without_parameters_produces_an_error(self):
        runner = CliRunner()
        result = runner.invoke(search_image)
        assert result.exit_code != 0  # should exit with an error code
        assert result.output.startswith(
            "Error: At least one of the search criteria has to be specified."
        )

    def test_rma_errors_are_reported(self, rma_all):
        error_msg = "Some error occurred"
        rma_all.side_effect = RMAError(error_msg)
        result = CliRunner().invoke(search_image, ["--id", "1"])
        assert result.exit_code != 0
        assert "error" in result.output
        assert error_msg in result.output

    def test_no_images_found(self, rma_all):
        rma_all.return_value = []
        result = CliRunner().invoke(search_image, ["--id", "1"])
        assert result.exit_code == 0
        assert "No images found" in result.output

    @pytest.mark.parametrize(
        ("cli_params", "expected_criteria"),
        (
            (["--id", "1"], {"id": "1"}),
            (["--dataset", "789"], {"data_set_id": "789"}),
            (["--specimen", "702694"], {"data_set": {"specimen_id": "702694"}}),
            (
                ["--gene-name", "my-gene"],
                {"data_set": {"genes": {"acronym": "my-gene"}}},
            ),
        ),
        ids=(
            "Filter by image ID",
            "Filter by dataset ID",
            "Filter by specimen ID",
            "Filter by gene acronym",
        ),
    )
    def test_search_filters(self, rma_all, cli_params, expected_criteria):
        """Test that CLI parameters are correctly translated to criteria."""
        result = CliRunner().invoke(search_image, cli_params)
        assert result.exit_code == 0
        assert rma_all.called_once
        # Get the args of the last call to rma_all
        (rma_parameters,), _kwargs = rma_all.call_args
        assert rma_parameters.criteria == expected_criteria

    @pytest.mark.parametrize(
        "command",
        [
            "--id",
            "--dataset",
        ],
    )
    def test_all_results_are_shown(self, rma_all, command):
        msg = [
            {
                "id": 1,
                "data_set_id": 1,
                "height": 100,
                "width": 100,
            },
            {
                "id": 2,
                "data_set_id": 1,
                "height": 200,
                "width": 200,
            },
            {
                "id": 3,
                "data_set_id": 1,
                "height": 300,
                "width": 300,
            },
        ]
        rma_all.return_value = msg
        runner = CliRunner()
        result = runner.invoke(search_image, [command, "whatever"])
        assert result.exit_code == 0

        # Check the output contains the correct number of bullet points
        assert len(re.findall(r"\*", result.output)) == len(msg)

        # Check each bullet point has the correct content
        for item in msg:
            assert re.search(
                (
                    fr"id: +{item['id']}, dataset: +{item['data_set_id']}, "
                    fr"h: +{item['height']}, w: +{item['width']}"
                ),
                result.output.strip(),
            )
