from click.testing import CliRunner

import atldld
from atldld.cli.info import info_group
from atldld.cli.root import root_cmd


class TestInfoSubgroup:
    def test_running_without_arguments_prints_help(self):
        runner = CliRunner()
        result = runner.invoke(root_cmd)
        assert result.exit_code == 0
        assert result.output.startswith("Usage:")

    def test_version_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info_group, ["version"])
        assert result.exit_code == 0
        assert atldld.__version__ in result.output

    def test_cache_command_works(self):
        runner = CliRunner()
        result = runner.invoke(info_group, ["cache"])
        assert result.exit_code == 0
        assert "atldld cache" in result.output.lower()

    def test_cache_command_shows_xdg(self, monkeypatch, tmpdir):
        runner = CliRunner()

        monkeypatch.delenv("XDG_CACHE_HOME")
        result = runner.invoke(info_group, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" not in result.output

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmpdir))
        result = runner.invoke(info_group, ["cache"])
        assert result.exit_code == 0
        assert "XDG_CACHE_HOME" in result.output
