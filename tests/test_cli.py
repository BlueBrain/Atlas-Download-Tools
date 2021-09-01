import pathlib
import re
import subprocess
import textwrap
from typing import Any, Dict

import click
import pytest
import responses
from click.testing import CliRunner

import atldld
from atldld.cli import (
    dataset,
    dataset_info,
    dataset_preview,
    get_dataset_meta_or_abort,
    info,
    root,
)


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
