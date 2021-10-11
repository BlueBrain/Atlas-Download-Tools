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

from atldld.cli.search import search_cmd, search_dataset
from atldld.requests import RMAError


class TestSearchSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(search_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")


class TestSearchDataset:
    @pytest.fixture
    def rma_all(self, mocker):
        return mocker.patch("atldld.requests.rma_all")

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

    def test_search_by_dataset_id(self, rma_all):
        rma_all.return_value = [
            {
                "id": 1,
                "plane_of_section_id": 1,
                "genes": [{"acronym": "Gad1"}],
                "section_images": [1, 2, 3],
            },
        ]
        runner = CliRunner()
        result = runner.invoke(search_dataset, ["--id", "1"])
        assert result.exit_code == 0
        assert re.search(
            r"id: +1, genes: +Gad1, coronal, 3 section images",
            result.output.strip(),
        )
        assert rma_all.called_once
        # First call, first arg
        rma_parameters = rma_all.call_args_list[0].args[0]
        assert rma_parameters.criteria == {"id": "1"}

    def test_search_by_specimen_id(self, rma_all):
        rma_all.return_value = [
            {
                "id": 1,
                "plane_of_section_id": 1,
                "genes": [{"acronym": "Gad1"}],
                "section_images": [1, 2, 3],
            },
        ]
        runner = CliRunner()
        result = runner.invoke(search_dataset, ["--specimen", "789"])
        assert result.exit_code == 0
        assert re.search(
            r"id: +1, genes: +Gad1, coronal, 3 section images",
            result.output.strip(),
        )
        assert rma_all.called_once
        # First call, first arg
        rma_parameters = rma_all.call_args_list[0].args[0]
        assert rma_parameters.criteria == {"specimen_id": "789"}
