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

import click


@click.group("download", help="Download dataset and section images")
def download_cmd():
    """Run download subcommands."""


@download_cmd.command(
    "dataset", help="Download and synchronize an entire section dataset"
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
def download_dataset(
    dataset_id,
    output_folder,
    downsample_ref,
    downsample_img,
    include_expression,
):
    """Download and synchronize dataset of section images."""
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
