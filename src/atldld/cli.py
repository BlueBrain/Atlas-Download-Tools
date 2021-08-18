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


@click.command(help="Location of the global cache folder.")
def cache_folder():
    """Print the location of the global cache folder."""
    from atldld.constants import GLOBAL_CACHE_FOLDER

    click.echo("Location of the global cache folder:")
    click.echo(str(GLOBAL_CACHE_FOLDER.resolve()))


root.add_command(info)
info.add_command(version)
info.add_command(cache_folder)


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


@click.command("preview", help="Plot a preview of dataset slices")
@click.argument("dataset_id", type=int)
def dataset_preview(dataset_id):
    import numpy as np
    from matplotlib.figure import Figure

    from atldld import constants, requests

    # Send request
    rma_parameters = requests.RMAParameters(
        "SectionDataSet",
        criteria={"id": dataset_id},
        include=("section_images",),
    )
    try:
        msg = requests.rma_all(rma_parameters)
    except requests.RMAError as exc:
        click.secho(
            f"An error occurred while querying the AIBS servers: {str(exc)}",
            fg="red",
        )

    # Check response
    if len(msg) == 0:
        click.secho(f"Dataset with ID {dataset_id} does not exist", fg="red")
        raise click.Abort
    elif len(msg) > 1:
        click.secho("Something went wrong: got more than one dataset", fg="red")
        raise click.Abort

    meta = msg[0]
    section_image_metas = meta.pop("section_images")
    section_image_metas.sort(key=lambda image_meta: image_meta["section_number"])
    section_image_metas = section_image_metas[:3]

    click.secho("Fetching the corner coordinates of the section images...", fg="green")
    all_corners = []
    with click.progressbar(section_image_metas) as progress:
        for image_meta in progress:
            corners = requests.get_ref_corners(
                image_meta["id"],
                image_meta["image_width"],
                image_meta["image_height"],
            )
            all_corners.append(np.array(corners))

    def draw_slice_2d(ax, points):
        coords = points.T
        ax.plot(*coords, color="green")
        ax.scatter(*coords, color="red")

    click.secho("Plotting...", fg="green")
    scale = 25
    n_p, n_i, n_r = np.array(constants.REF_DIM_1UM) / scale

    fig = Figure(figsize=(14, 4), dpi=200)
    axs = fig.subplots(
        ncols=4,
        sharey=True,
        gridspec_kw={"width_ratios": [16 / 7, 1, 16 / 7, 1]}
    )
    fig.suptitle(f"Dataset ID {dataset_id}", fontsize=32)
    for ax in axs.ravel():
        ax.grid(True, linestyle=":", color="gray")
        ax.set_ylim((0, n_r))
    #     ax.set_aspect("equal")
    ax1, ax2, ax3, ax4 = axs.ravel()

    ax1.set_title("Edge $(0, 0)-(1, 0)$")
    ax1.set_xlabel("p (coronal)", fontsize=16)
    ax1.set_ylabel("r (sagittal)", fontsize=16)
    ax1.axvline(0, color="blue", linestyle=":")
    ax1.axvline(n_p, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax1, corners[np.ix_([0, 1], [0, 2])] / scale)

    ax2.set_title("Edge $(1, 0)-(1, 1)$")
    ax2.set_xlabel("i (transversal)", fontsize=16)
    ax2.axvline(0, color="blue", linestyle=":")
    ax2.axvline(n_i, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax2, corners[np.ix_([1, 2], [1, 2])] / scale)

    ax3.set_title("Edge $(1, 1)-(0, 1)$")
    ax3.set_xlabel("p (coronal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax3, corners[np.ix_([2, 3], [0, 2])] / scale)
    ax3.axvline(0, color="blue", linestyle=":")
    ax3.axvline(n_p, color="blue", linestyle=":")
    ax3.invert_xaxis()

    ax4.set_title("Edge $(0, 1)-(0, 0)$")
    ax4.set_xlabel("i (transversal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax4, corners[np.ix_([3, 0], [1, 2])] / scale)
    ax4.axvline(0, color="blue", linestyle=":")
    ax4.axvline(n_i, color="blue", linestyle=":")
    ax4.invert_xaxis()
    fig.tight_layout()
    fig.savefig(f"dataset-id-{dataset_id}-preview.png")


root.add_command(dataset)
dataset.add_command(dataset_info)
dataset.add_command(dataset_preview)
