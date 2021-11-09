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
@click.option("-i", "--id", "dataset_id", help="The dataset ID")
@click.option("-g", "--gene-name", "gene_acronym", help="The gene acronym")
@click.option("-s", "--specimen", "specimen_id", help="The specimen ID")
@click.option(
    "-p",
    "--plane-of-section",
    "plane_of_section",
    help="The plane of section, e.g. coronal, sagittal, etc.",
)
def search_dataset(dataset_id, specimen_id, gene_acronym, plane_of_section):
    """Run search subcommand."""
    from atldld import requests
    from atldld.dataset import PlaneOfSection

    criteria = {}
    if dataset_id is not None:
        criteria["id"] = dataset_id
    if specimen_id is not None:
        criteria["specimen_id"] = specimen_id
    if gene_acronym is not None:
        criteria["genes"] = {"acronym": gene_acronym}
    if plane_of_section is not None:
        criteria["plane_of_section"] = {"name": plane_of_section}

        if plane_of_section not in PlaneOfSection.__names__.values():
            click.secho(
                f'Unknown plane of section name: "{plane_of_section}"', fg="yellow"
            )
            click.secho("Probably no datasets will be found.", fg="yellow")
            click.secho("Valid plane of section names are:", fg="yellow")
            for name in PlaneOfSection.__names__.values():
                click.secho(f"* {name}", fg="yellow")

    if len(criteria) == 0:
        raise click.ClickException(
            "At least one of the search criteria has to be specified. "
            "Use the --help flag to see all available criteria."
        )

    # Send request
    rma_parameters = requests.RMAParameters(
        "SectionDataSet",
        criteria=criteria,
        include=["genes", "section_images"],
    )
    click.secho("Searching...", fg="green")
    try:
        msg = requests.rma_all(rma_parameters)
    except requests.RMAError as exc:
        raise click.ClickException(
            f"An error occurred while querying the AIBS servers: {str(exc)}"
        )

    if len(msg) == 0:
        click.secho("No datasets found", fg="red")
    else:
        click.secho(f"{len(msg)} dataset(s) found:", fg="green")
        for meta in msg:
            genes = ", ".join(gene["acronym"] for gene in meta["genes"])
            plane_of_section = PlaneOfSection(meta["plane_of_section_id"])
            click.echo(
                f"* id: {meta['id']:10d}, genes: {genes:>10s}, "
                f"{plane_of_section}, {len(meta['section_images'])} section images"
            )


@search_cmd.command("image", help="Search section images")
@click.option("-i", "--id", "image_id", help="The image ID")
@click.option("-d", "--dataset", "dataset_id", help="The dataset ID")
@click.option("-g", "--gene-name", "gene_acronym", help="The gene acronym")
@click.option("-s", "--specimen", "specimen_id", help="The specimen ID")
def search_image(image_id, dataset_id, gene_acronym, specimen_id):
    """Run search subcommand."""
    from collections import defaultdict
    from typing import Any, DefaultDict

    from atldld import requests

    criteria: DefaultDict[str, Any] = defaultdict(dict)
    if image_id is not None:
        criteria["id"] = image_id
    if dataset_id is not None:
        criteria["data_set_id"] = dataset_id
    if gene_acronym is not None:
        criteria["data_set"]["genes"] = {"acronym": gene_acronym}
    if specimen_id is not None:
        criteria["data_set"]["specimen_id"] = specimen_id

    if len(criteria) == 0:
        raise click.ClickException(
            "At least one of the search criteria has to be specified. "
            "Use the --help flag to see all available criteria."
        )

    # Send request
    rma_parameters = requests.RMAParameters(
        "SectionImage",
        criteria=criteria,
    )
    click.secho("Searching...", fg="green")
    try:
        msg = requests.rma_all(rma_parameters)
    except requests.RMAError as exc:
        raise click.ClickException(
            f"An error occurred while querying the AIBS servers: {str(exc)}"
        )

    if len(msg) == 0:
        click.secho("No images found", fg="red")
    else:
        click.secho(f"{len(msg)} image(s) found:", fg="green")
        for meta in msg:
            click.echo(
                f"* id: {meta['id']:10d}, dataset: {meta['data_set_id']:>10d}, "
                f"h: {meta['height']:>5d}, w: {meta['width']:>5d}"
            )
