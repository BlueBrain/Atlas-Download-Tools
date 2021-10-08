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
"""Implementation of the "atldld dataset" subcommand."""
import pathlib
from typing import Any, Dict, Optional, Sequence

import click


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


@click.group("dataset", help="Commands related to atlas datasets")
def dataset_cmd():
    """Run dataset subcommands."""


@dataset_cmd.command("info", help="Get information for a given dataset ID")
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


@dataset_cmd.command(
    "download", help="Download and synchronize an entire section dataset"
)
@click.argument("dataset_id", type=str)
@click.argument(
    "output_folder",
    type=click.Path(exists=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option(
    "--downsample-ref",
    type=int,
    default=25,
    show_default=True,
    help="Downsampling coefficient for the reference space. Determines the size "
    "of the synchronized image.",
)
@click.option(
    "--downsample-img",
    type=int,
    default=0,
    show_default=True,
    help="Downsampling coefficient for the image download.",
)
@click.option(
    "-e",
    "--include-expression",
    is_flag=True,
    help="Include expression image.",
)
def dataset_download(
    dataset_id,
    output_folder,
    downsample_ref,
    downsample_img,
    include_expression,
):
    """Download and synchronize an entire section dataset."""
    import json
    import textwrap

    from PIL import Image

    from atldld.sync import DatasetDownloader, DatasetNotFoundError

    # Prepare paths
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    metadata_path = output_folder / "metadata.json"

    downloader = DatasetDownloader(
        dataset_id,
        downsample_ref=downsample_ref,
        downsample_img=downsample_img,
        include_expression=include_expression,
    )
    cli_input = f"""
    Dataset ID               : {dataset_id}
    Downsample reference     : {downsample_ref}
    Downsample image         : {downsample_img}
    Include expression       : {include_expression}
    Output folder            : {output_folder}
    """
    click.secho(textwrap.dedent(cli_input).strip(), fg="blue")

    try:
        downloader.fetch_metadata()
    except DatasetNotFoundError as exc:
        raise click.ClickException(str(exc))
    n_images = len(downloader)

    additional_info = f"""
    Number of section images : {n_images}
    Section thickness        : {downloader.metadata["dataset"]["section_thickness"]}µm
    Plane of section         : {downloader.metadata["dataset"]["plane_of_section_id"]}
    """
    click.secho(textwrap.dedent(additional_info).strip(), fg="green")

    metadata = {
        "dataset_id": dataset_id,
        "downsample_ref": downsample_ref,
        "downsample_img": downsample_img,
        "plane_of_section": downloader.metadata["dataset"]["plane_of_section_id"],
        "section_thickness": downloader.metadata["dataset"]["section_thickness"],
        "per_image": {},
    }

    with click.progressbar(downloader.run(), length=n_images) as progress:
        for image_id, section_coordinate, img, img_expr, df in progress:
            img_synced = df.warp(img, c=img[0, 0].tolist())

            img_path = output_folder / f"{image_id}.png"
            Image.fromarray(img_synced, mode="RGB").save(img_path)

            if img_expr is not None:
                img_expr_synced = df.warp(img_expr)
                img_expr_path = output_folder / f"{image_id}_expr.png"
                Image.fromarray(img_expr_synced, mode="RGB").save(img_expr_path)

            metadata["per_image"][image_id] = {
                "section_coordinate": section_coordinate,
                "section_coordinate_scaled": section_coordinate
                / metadata["section_thickness"],
            }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)


@dataset_cmd.command("preview", help="Plot a preview of dataset slices")
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
