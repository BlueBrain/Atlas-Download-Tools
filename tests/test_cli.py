import re
import subprocess
import textwrap
from typing import Any, Dict

import pytest
import responses
from click.testing import CliRunner

import atldld
from atldld.cli import dataset, dataset_info, info, root


def test_cli_entrypoint_is_installed():
    subprocess.check_call("atldld")


class TestCliRoot:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(root)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")

    @pytest.mark.parametrize("subgroup", ("info", "dataset"))
    def test_subgroup_installed(self, subgroup):
        runner = CliRunner()
        result = runner.invoke(root, [subgroup])
        assert result.exit_code == 0


class TestInfoSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(info)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")

    def test_version_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info, ["version"])
        assert result.exit_code == 0
        assert atldld.__version__ in result.output

    def test_cache_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info, ["cache"])
        assert result.exit_code == 0
        assert "atldld cache" in result.output.lower()

    def test_cache_command_shows_xdg(self, monkeypatch, tmpdir):
        runner = CliRunner()

        monkeypatch.delenv("XDG_CACHE_HOME")
        result = runner.invoke(info, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" not in result.output

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmpdir))
        result = runner.invoke(info, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" in result.output



class TestDatasetSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(dataset)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")


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

    @responses.activate
    def test_rma_errors_are_caught(self):
        # this should lead to an RMAError
        response_json = {"success": False, "msg": "Some error"}
        responses.add(responses.GET, re.compile(""), json=response_json)
        runner = CliRunner()
        result = runner.invoke(dataset_info, ["111"])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # means exception was caught
        assert "an error occurred" in result.output.lower()

    @responses.activate
    def test_invalid_dataset_id_is_reported(self):
        response_json = {
            "success": True,
            "start_row": 0,
            "num_rows": 0,
            "total_rows": 0,
            "msg": [],
        }
        responses.add(responses.GET, re.compile(""), json=response_json)
        runner = CliRunner()
        result = runner.invoke(dataset_info, ["0"])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # means exception was caught
        assert "does not exist" in result.output.lower()

    @responses.activate
    def test_multiple_datasets_returned_is_reported(self):
        response_json = {
            "success": True,
            "start_row": 0,
            "num_rows": 2,
            "total_rows": 2,
            "msg": ["dataset1", "dataset2"],
        }
        responses.add(responses.GET, re.compile(""), json=response_json)
        runner = CliRunner()
        result = runner.invoke(dataset_info, ["0"])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # means exception was caught
        assert "more than one dataset" in result.output.lower()
