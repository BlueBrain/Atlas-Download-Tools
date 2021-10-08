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
"""Implementation of the "atldld search" subcommand."""
import click


@click.group("search", help="Search datasets and section images")
def search_cmd():
    """Run search subcommand."""


@search_cmd.command("dataset", help="Search datasets")
def search_dataset_cmd():
    """Run search subcommand."""


@search_cmd.command("img", help="Search section images")
def search_img_cmd():
    """Run search subcommand."""
