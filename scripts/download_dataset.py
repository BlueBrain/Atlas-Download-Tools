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
"""Download and postprocess ISH dataset from Allen Brain."""
import argparse
import pathlib
import sys

import numpy as np
import PIL


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("gene_name")
    args = parser.parse_args()

    return args


def postprocess_dataset(dataset):
    """Post process given dataset.

    Parameters
    ----------
    dataset : iterable
        List containing dataset, each element being tuple
        (img_id, section_coordinate, img, df)

    Returns
    -------
    dataset_np : np.ndarray
        Array containing gene expressions of the dataset.
    metadata_dict : dict
        Dictionary containing metadata of the dataset.
        Keys are section numbers and image ids.
    """
    metadata_dict = {}
    section_numbers = []
    image_ids = []
    dataset_np = []

    for img_id, section_coordinate, img, df in dataset:
        if section_coordinate is None:
            # In `atldld.sync.download_dataset` if there is a problem during the download
            # the generator returns `(img_id, None, None, None, None)
            # TODO: maybe notify the user somehow?
            continue

        section_numbers.append(section_coordinate // 25)
        image_ids.append(img_id)
        warped_img = df.warp(img, border_mode="constant", c=img[0, 0, :].tolist())
        dataset_np.append(warped_img)

    dataset_np = np.array(dataset_np)

    metadata_dict["section_numbers"] = section_numbers
    metadata_dict["image_ids"] = image_ids
    metadata_dict["image_shape"] = warped_img.shape

    return dataset_np, metadata_dict


def main():
    """Download gene expression dataset."""
    # Imports
    import json

    from atldld.sync import download_dataset
    from atldld.utils import CommonQueries, get_experiment_list_from_gene

    args = parse_args()
    gene = args.gene_name
    # To avoid Decompression Warning
    PIL.Image.MAX_IMAGE_PIXELS = 200000000

    # Download dataset on allen
    for axis in ['sagittal', 'coronal']:

        file_dir = pathlib.Path(f"{axis}/{gene}/")
        if not file_dir.exists():
            file_dir.mkdir(parents=True)

        experiment_list = get_experiment_list_from_gene(gene, axis)
        for experiment_id in experiment_list:
            dataset = download_dataset(experiment_id)
            axis = CommonQueries.get_axis(experiment_id)
            dataset_np, metadata_dict = postprocess_dataset(dataset)
            metadata_dict["axis"] = axis

            np.save(file_dir / f"{experiment_id}.npy", dataset_np)
            with open(file_dir / f"{experiment_id}.json", 'w') as f:
                json.dump(metadata_dict, f)


if __name__ == "__main__":
    sys.exit(main())
