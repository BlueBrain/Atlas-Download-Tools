import subprocess

import pytest
from click.testing import CliRunner

from atldld.cli.root import root_cmd


def test_cli_entrypoint_is_installed():
    subprocess.check_call("atldld")


class TestCliRoot:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(root_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")

    @pytest.mark.parametrize("subgroup", ("info", "dataset"))
    def test_subgroup_installed(self, subgroup):
        runner = CliRunner()
        result = runner.invoke(root_cmd, [subgroup])
        assert result.exit_code == 0
