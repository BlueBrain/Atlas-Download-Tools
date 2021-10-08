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
from click.testing import CliRunner

import atldld
from atldld.cli.info import info_cmd
from atldld.cli.root import root_cmd


class TestInfoSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(root_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")

    def test_version_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info_cmd, ["version"])
        assert result.exit_code == 0
        assert atldld.__version__ in result.output

    def test_cache_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info_cmd, ["cache"])
        assert result.exit_code == 0
        assert "atldld cache" in result.output.lower()

    def test_cache_command_shows_xdg(self, monkeypatch, tmpdir):
        runner = CliRunner()

        monkeypatch.delenv("XDG_CACHE_HOME")
        result = runner.invoke(info_cmd, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" not in result.output

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmpdir))
        result = runner.invoke(info_cmd, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" in result.output
