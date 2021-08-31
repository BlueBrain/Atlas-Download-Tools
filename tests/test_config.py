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
import pathlib

import pytest

from atldld.config import user_cache_dir


class TestUserCacheDir:
    def test_default_cache_dir_works(self):
        cache_dir = user_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_default_cache_in_user_home(self, monkeypatch):
        # All tests use a custom cache directory set by the custom_cache_dir
        # fixture in conftest. Unset this just for this test.
        monkeypatch.delenv("XDG_CACHE_HOME")
        cache_dir = user_cache_dir(create=False)
        assert pathlib.Path.home() in cache_dir.parents

    @pytest.mark.parametrize("create", [True, False])
    def test_custom_cache_dir_works(self, monkeypatch, tmpdir, create):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmpdir))
        cache_dir = user_cache_dir(create=create)
        assert cache_dir == pathlib.Path(tmpdir) / "atldld"
        if create:
            assert cache_dir.exists()
            assert cache_dir.is_dir()
