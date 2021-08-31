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
"""Various constants."""
import appdirs
import os
import pathlib

# Set the cache folder
if "XDG_CACHE_HOME" in os.environ:
    # appdirs reads XDG_CACHE_HOME only on Linux. Make it work for macOS
    GLOBAL_CACHE_FOLDER = pathlib.Path(os.getenv("XDG_CACHE_HOME")) / "atldld"
else:
    GLOBAL_CACHE_FOLDER = pathlib.Path(appdirs.user_cache_dir("atldld"))
GLOBAL_CACHE_FOLDER = GLOBAL_CACHE_FOLDER.resolve()
GLOBAL_CACHE_FOLDER.mkdir(exist_ok=True, parents=True)
