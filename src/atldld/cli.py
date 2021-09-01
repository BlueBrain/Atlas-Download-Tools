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
from typing import Any, Dict, Optional, Sequence

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
    name="cache",
    help="""
    Location of the atldld cache directory.

    By default it is configured as a subdirectory of the OS-specific cache
    directory. If the XDG_CACHE_HOME environment variable is set its value will
    override the OS-specific cache directory.
    """,
)
def cache_dir():
    """Print the location of the global cache directory."""
    import os

    from atldld.config import user_cache_dir

    if "XDG_CACHE_HOME" in os.environ:
        suffix = " (configured via XDG_CACHE_HOME)"
    else:
        suffix = ""
    click.secho(f"Location of the atldld cache{suffix}:", fg="green")
    click.echo(str(user_cache_dir(create=False).resolve().as_uri()))


root.add_command(info)
info.add_command(version)
info.add_command(cache_dir)


# ============================= Dataset subcommand =============================
def get_dataset_meta_or_abort(
    dataset_id: int, include: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Download the dataset metadata.

    Parameters
    ----------
    dataset_id
        The dataset ID.
    include
        The include keys to use in the RMA query.

    Returns
    -------
    meta : dict
        The dataset metadata.

    Raises
    ------
    click.Abort
        Whenever the metadata download fails or yields unexpected results.
    """
    from atldld import requests

    # Send request
    rma_parameters = requests.RMAParameters(
        "SectionDataSet",
        criteria={"id": dataset_id},
        include=include,
    )
    try:
        msg = requests.rma_all(rma_parameters)
    except requests.RMAError as exc:
        raise click.ClickException(
            f"An error occurred while querying the AIBS servers: {str(exc)}"
        )

    # Check response
    if len(msg) == 0:
        raise click.ClickException(f"Dataset with ID {dataset_id} does not exist")
    elif len(msg) > 1:
        raise click.ClickException("Something went wrong: got more than one dataset")

    meta = msg[0]
    if not isinstance(meta, dict) or not all(isinstance(key, str) for key in meta):
        raise click.ClickException("Got an unexpected dataset information format")

    return meta


@click.group(help="Commands related to atlas datasets")
def dataset():
    """Run dataset subcommands."""


@click.command("info", help="Get information for a given dataset ID")
@click.argument("dataset_id", type=int)
def dataset_info(dataset_id):
    """Get information for a given dataset ID."""
    import textwrap

    import atldld.dataset

    # Download the dataset metadata
    meta = get_dataset_meta_or_abort(dataset_id, include=["genes", "section_images"])

    # Print response
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


@click.command("preview", help="Plot a preview of dataset slices")
@click.argument("dataset_id", type=int)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
    help="""
    The output directory for the plot figure. If not provided the current
    working directory will be used.
    """,
)
def dataset_preview(dataset_id, output_dir):
    """Plot a sketch of section images mapped into the reference space."""
    import pathlib

    import atldld.dataset
    import atldld.utils
    from atldld import plot

    # Download the dataset metadata
    meta = get_dataset_meta_or_abort(dataset_id, include=["section_images"])
    plane_of_section = atldld.dataset.PlaneOfSection(meta["plane_of_section_id"])
    section_image_metas = meta.pop("section_images")
    section_image_metas.sort(key=lambda image_meta_: image_meta_["section_number"])

    click.secho("Fetching the corner coordinates of the section images...", fg="green")
    all_corners = []
    with click.progressbar(section_image_metas) as progress:
        for image_meta in progress:
            corners = atldld.utils.get_corners_in_ref_space(
                image_meta["id"],
                image_meta["image_width"],
                image_meta["image_height"],
            )
            all_corners.append(corners)

    click.secho("Plotting...", fg="green")
    img_file_name = f"dataset-id-{dataset_id}-preview.png"
    if output_dir is None:
        img_path = pathlib.Path.cwd() / img_file_name
    else:
        img_path = pathlib.Path(output_dir) / img_file_name
        img_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plot.dataset_preview(all_corners, plane_of_section)
    fig.suptitle(f"Dataset ID {dataset_id}", fontsize=32)
    fig.set_dpi(200)
    fig.savefig(img_path)
    click.secho("Figure was saved in ", fg="green", nl=False)
    click.secho(f"{img_path.resolve().as_uri()}", fg="yellow", bold=True)


root.add_command(dataset)
dataset.add_command(dataset_info)
dataset.add_command(dataset_preview)
