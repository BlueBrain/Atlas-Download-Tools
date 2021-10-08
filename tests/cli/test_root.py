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
