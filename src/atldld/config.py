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
"""Configuration of the atldld CLI app and library."""
import os
import pathlib

import appdirs


def user_cache_dir(create: bool = True) -> pathlib.Path:
    """Determine currently configured cache directory for atldld.

    Parameters
    ----------
    create
        If true the returned directory will be guaranteed to exist.

    Returns
    -------
    pathlib.Path
        The currently configured cache directory for atldld.
    """
    app_name = "atldld"
    if "XDG_CACHE_HOME" in os.environ:
        # appdirs reads XDG_CACHE_HOME only on Linux. Make it work for macOS too
        cache_dir = pathlib.Path(os.environ["XDG_CACHE_HOME"]) / app_name
    else:
        cache_dir = pathlib.Path(appdirs.user_cache_dir(app_name))
    cache_dir = cache_dir.resolve()
    if create:
        cache_dir.mkdir(exist_ok=True, parents=True)

    return cache_dir
