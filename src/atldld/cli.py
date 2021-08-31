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
"""The command line interface (CLI) for Atlas-Download-Tools."""
import click

import atldld


@click.group(
    help="""\
    Atlas-Download-Tools command line interface.

    For detailed instructions see the documentation of the corresponding sub-commands.
    """
)
def root():
    """Run the command line interface for Atlas-Download-Tools."""


# ============================= Info subcommand ================================


@click.group(help="Informational subcommands.")
def info():
    """Run informational subcommands."""


@click.command(help="Version of Atlas-Download-Tools.")
def version():
    """Print the version of Atlas-Download-Tools."""
    click.echo(f"Atlas-Download-Tools version {atldld.__version__}")


@click.command(
    help="""
    Location of the atldld cache directory.

    By default it is configured as a subdirectory of the OS-specific cache
    directory. If the XDG_CACHE_HOME environment variable is set its value will
    override the OS-specific cache directory.
    """,
)
def cache_dir():
    """Print the location of the global cache directory."""
    from atldld.constants import user_cache_dir

    click.echo("The atldld cache directory is:")
    click.echo(str(user_cache_dir(create=False).resolve().as_uri()))


root.add_command(info)
info.add_command(version)
info.add_command(cache_dir)


# ============================= Dataset subcommand =============================


@click.group(help="Commands related to atlas datasets")
def dataset():
    """Run dataset subcommands."""


@click.command("info", help="Get information for a given dataset ID")
@click.argument("dataset_id", type=int)
def dataset_info(dataset_id):
    """Get information for a given dataset ID."""
    import textwrap

    import atldld.dataset
    from atldld import requests

    # Send request
    rma_parameters = requests.RMAParameters(
        "SectionDataSet",
        criteria={"id": dataset_id},
        include=("genes", "section_images"),
    )
    try:
        msg = requests.rma_all(rma_parameters)
    except requests.RMAError as exc:
        click.secho(
            f"An error occurred while querying the AIBS servers: {str(exc)}",
            fg="red",
        )
        raise click.Abort

    # Check response
    if len(msg) == 0:
        click.secho(f"Dataset with ID {dataset_id} does not exist", fg="red")
        raise click.Abort
    elif len(msg) > 1:
        click.secho("Something went wrong: got more than one dataset", fg="red")
        raise click.Abort

    # Print response
    meta = msg[0]
    section_images = meta.pop("section_images")
    r_str = meta["red_channel"] or "-"
    g_str = meta["green_channel"] or "-"
    b_str = meta["blue_channel"] or "-"
    plane_of_section = atldld.dataset.PlaneOfSection(meta["plane_of_section_id"])
    reference_space = atldld.dataset.ReferenceSpace(meta["reference_space_id"])
    output = f"""
    ID                       : {meta["id"]}
    Sphinx ID                : {meta["sphinx_id"]}
    Specimen ID              : {meta["specimen_id"]}
    Name                     : {meta["name"] or "-"}
    Failed                   : {"Yes" if meta["failed"] else "No"}
    Expression               : {"Yes" if meta["expression"] else "No"}
    Gene(s)                  : {", ".join(gene["acronym"] for gene in meta["genes"])}
    RGB channels             : {r_str} / {g_str} / {b_str}
    Section thickness        : {meta["section_thickness"]}µm
    Plane of section         : {plane_of_section}
    Number of section images : {len(section_images)}
    Reference space          : {reference_space.value} ({reference_space})
    """
    click.secho(textwrap.dedent(output).strip(), fg="green")


root.add_command(dataset)
dataset.add_command(dataset_info)
