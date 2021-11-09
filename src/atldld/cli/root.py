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
"""Implementation of the "atldld" command line entrypoint.

We follow a particular naming for the functions that implement the CLI commands.
Given the command `atldld group command` the function representing the `group`
subcommand shall be named `group_cmd` and the function that implements `command`
shall be named `group_command`. Both `group_cmd` and `group_command` shall be
placed in the submodule `atldld.cli.group`.

For example, the `atldld dataset download` command leads to the following
functions:
* `atldld.cli.dataset::dataset_cmd` (subcommand group)
* `atldld.cli.dataset::dataset_download` (`download` command implementation)
"""
import click

from atldld.cli.dataset import dataset_cmd
from atldld.cli.download import download_cmd
from atldld.cli.info import info_cmd
from atldld.cli.search import search_cmd


@click.group(
    help="""\
    Atlas-Download-Tools command line interface.

    For detailed instructions see the documentation of the corresponding sub-commands.
    """
)
def root_cmd():
    """Run the command line interface for Atlas-Download-Tools."""


root_cmd.add_command(dataset_cmd)
root_cmd.add_command(download_cmd)
root_cmd.add_command(info_cmd)
root_cmd.add_command(search_cmd)
