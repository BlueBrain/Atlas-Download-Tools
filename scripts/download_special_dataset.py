# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download special dataset (with 2 genes) from Allen Brain."""
import argparse
import pathlib
import sys

import requests


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()

    return args


def allen_dataset_info(dataset_id):
    """Extract dataset info from Allen API.

    Parameters
    ----------
    dataset_id: int or str
        Dataset ID from Allen Brain API

    Returns
    -------
    dataset_info: dict
        Dictionary from Allen Brain containing
        some metadata for the given dataset.
    """
    url = "http://api.brain-map.org/api/v2/data/query.json?"
    url += f"criteria=model::SectionDataSet,rma::criteria,[id$eq{dataset_id}]"
    r = requests.get(url)
    raw = r.json()
    dataset_info = raw["msg"][0]
    return dataset_info


def main():
    """Download gene expression dataset."""
    # Imports
    import json

    import numpy as np
    import PIL
    from atldld.sync import download_dataset

    args = parse_args()
    # To avoid DecompressionBombError --> highest value from 3 datasets we are using
    PIL.Image.MAX_IMAGE_PIXELS = 192491520

    # Download dataset on allen
    dataset_id = args.dataset_id
    dataset_info = allen_dataset_info(dataset_id)
    dataset_info["dataset_id"] = dataset_id
    dataset = list(download_dataset(dataset_id))
    dataset_info["num_images"] = len(dataset)
    # dataset = sequence of (image_id, p, img, df)

    # Create volume with gene expression
    p_values = [data[1] for data in dataset]
    dataset_info["original_p_values"] = p_values
    p_values = [p // 25 for p in p_values]

    image_ids = [int(data[0]) for data in dataset]
    dataset_info["image_ids"] = image_ids
    # Compute the shape of the output volume based on the shape of
    # the first slice and the highest p-value
    _, _, img_0, df_0 = dataset[0]
    img_0_reg = df_0.warp(img_0)
    shape = (int(max(p_values) + 1), *img_0_reg.shape[:2])

    red_gene = np.zeros(shape)
    green_gene = np.zeros(shape)
    blue_gene = np.zeros(shape)
    all_dfs = []
    for _, p, img, df in dataset:
        section_number = int(p // 25)
        img_reg = df.warp(img)
        red_gene[section_number, :, :] = img_reg[:, :, 0]
        green_gene[section_number, :, :] = img_reg[:, :, 1]
        blue_gene[section_number, :, :] = img_reg[:, :, 2]
        all_dfs.append(df)

    output_dir = pathlib.Path("special_volumes") / f"dataset_{dataset_id}"
    output_dir.mkdir(parents=True)
    np.save(output_dir / dataset_info["red_channel"], red_gene)
    np.save(output_dir / dataset_info["green_channel"], green_gene)
    np.save(output_dir / dataset_info["blue_channel"], blue_gene)
    np.save(output_dir / "dfs.npy", np.stack(all_dfs))
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(dataset_info, f)


if __name__ == "__main__":
    sys.exit(main())
