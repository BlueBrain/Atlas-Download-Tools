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
"""Implementation of the "atldld info" subcommand."""
import click

import atldld


@click.group("info", help="Informational subcommands.")
def info_cmd():
    """Run informational subcommands."""


@info_cmd.command("version", help="Version of Atlas-Download-Tools.")
def info_version():
    """Print the version of Atlas-Download-Tools."""
    click.echo(f"Atlas-Download-Tools version {atldld.__version__}")


@info_cmd.command(
    name="cache",
    help="""
    Location of the atldld cache directory.

    By default it is configured as a subdirectory of the OS-specific cache
    directory. If the XDG_CACHE_HOME environment variable is set its value will
    override the OS-specific cache directory.
    """,
)
def info_cache():
    """Print the location of the global cache directory."""
    import os

    from atldld.config import user_cache_dir

    if "XDG_CACHE_HOME" in os.environ:
        suffix = " (configured via XDG_CACHE_HOME)"
    else:
        suffix = ""
    click.secho(f"Location of the atldld cache{suffix}:", fg="green")
    click.echo(str(user_cache_dir(create=False).resolve().as_uri()))
