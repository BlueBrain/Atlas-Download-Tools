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
"""A collection of synchronizations related to working with Allen's API.

Notes
-----
See the module `atldld.utils.py` for lower level
functions that are called within this module.
"""

from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from atldld.base import DisplacementField
from atldld.constants import AFFINE_TEMPLATES
from atldld.requests import RMAParameters, rma_all
from atldld.utils import get_image


class DatasetNotFoundError(Exception):
    """Raised when there is no dataset for the given dataset ID."""


def xy_to_pir(
    coords_img: np.ndarray,
    affine_2d: np.ndarray,
    affine_3d: np.ndarray,
) -> np.ndarray:
    """Transform coordinates from the image space to the reference space.

    Parameters
    ----------
    coords_img
        Array of shape `(3, N)` where the first axis contains the
        `x`, `y` and `section_number * section_thickness`. Note that
        both the `section_number` (image specific) and
        `section_thickness` (dataset specific) can be retrieved from the
        Allen Brain API.
    affine_2d
        Matrix of shape `(2, 3)` representing a 2D affine transformation. It
        can be retrieved from the section image metadata via the
        Allen Brain API. More specifically, it is stored under the
        `tsv_**` entries.
    affine_3d
        Matrix of shape `(3, 4)` representing a 3D affine transformation. It
        can be retrieved from the dataset metadata via the Allen Brain API.
        More specifically, it is stored under the `tvr_**` entries.

    Returns
    -------
    coords_ref : np.ndarray
        Array of shape `(3, N)` where the first axis contains the `p`, `i`,
        `r` coordinates.
    """
    dtype = np.float32
    n_coords = coords_img.shape[1]

    coords_img_ = np.concatenate([coords_img, np.ones((1, n_coords), dtype=dtype)])

    affine_2d_ = np.array(
        [
            [affine_2d[0, 0], affine_2d[0, 1], 0, affine_2d[0, 2]],
            [affine_2d[1, 0], affine_2d[1, 1], 0, affine_2d[1, 2]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )
    affine_3d_ = affine_3d.astype(dtype)
    coords_ref = (affine_3d_ @ affine_2d_) @ coords_img_

    return coords_ref


def pir_to_xy(
    coords_ref: np.ndarray,
    affine_2d: np.ndarray,
    affine_3d: np.ndarray,
) -> np.ndarray:
    """Transform coordinates from the reference space to the image space.

    Parameters
    ----------
    coords_ref
        Array of shape `(3, N)` where the first axis contains the
        `p`, `i` and `r`.
    affine_2d
        Matrix of shape `(2, 3)` representing a 2D affine transformation. It
        can be retrieved from the section image metadata via the
        Allen Brain API. More specifically, it is stored under the
        `tvs_**` entries.
    affine_3d
        Matrix of shape `(3, 4)` representing a 3D affine transformation. It
        can be retrieved from the dataset metadata via the Allen Brain API.
        More specifically, it is stored under the `trv_**` entries.

    Returns
    -------
    coords_img : np.ndarray
        Array of shape `(3, N)` where the first axis contains the `x`, `y`,
        `section_number * section_thickness`. Note that the `section_number`
        can be an arbitrary float that is most likely not equal to a
        `section_number` of any section image.
    """
    dtype = np.float32

    n_coords = coords_ref.shape[1]
    coords_ref_ = np.concatenate([coords_ref, np.ones((1, n_coords), dtype=dtype)])

    affine_2d_ = np.array(
        [
            [affine_2d[0, 0], affine_2d[0, 1], 0, affine_2d[0, 2]],
            [affine_2d[1, 0], affine_2d[1, 1], 0, affine_2d[1, 2]],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )
    affine_3d_ = np.concatenate(
        [
            affine_3d,
            np.array([[0, 0, 0, 1]]),
        ],
        axis=0,
    ).astype(dtype)

    coords_img = (affine_2d_ @ affine_3d_) @ coords_ref_

    return coords_img


def get_parallel_transform(
    slice_coordinate: float,
    affine_2d: np.ndarray,
    affine_3d: np.ndarray,
    axis: str = "coronal",
    downsample_ref: int = 1,
    downsample_img: int = 0,
) -> DisplacementField:
    """Compute displacement field between the reference space and the image.

    Parameters
    ----------
    slice_coordinate
        Value of the `axis` coordinate at which the image was sliced.
    affine_2d
        Matrix of shape `(2, 3)` representing a 2D affine transformation.
    affine_3d
        Matrix of shape `(3, 4)` representing a 3D affine transformation.
    axis : str, {"coronal", "sagittal", "transverse"}
        Axis along which the slice was made.
    downsample_ref
        Downscaling of the reference space grid. If set to 1 no
        downsampling takes place. The higher the value the smaller the grid
        in the reference space and the faster the matrix multiplication.
    downsample_img
        The downloaded image will have both the height and the width
        downsampled by `2 ** downsample_img`.

    Returns
    -------
    DisplacementField
        Displacement field representing the transformation between the
        reference space and the image. Note that one can directly use it
        to register raw histological images to the reference space.
    """
    dtype = np.float32

    refspace = (  # order matters
        ("coronal", 13200),
        ("transverse", 8000),
        ("sagittal", 11400),
    )

    axis_fixed = [i for i, a in enumerate(refspace) if a[0] == axis][0]
    axes_variable = [i for i, a in enumerate(refspace) if a[0] != axis]

    grid_shape = [refspace[i][1] // downsample_ref for i in axes_variable]
    n_pixels = np.prod(grid_shape)
    grid = np.indices(grid_shape, dtype=dtype)

    coords_ref = np.ones((3, n_pixels), dtype=dtype)
    coords_ref[axis_fixed] *= slice_coordinate
    coords_ref[axes_variable] = grid.reshape(2, n_pixels) * downsample_ref

    coords_img = pir_to_xy(coords_ref, affine_2d, affine_3d)[:2]

    tx = coords_img[0].reshape(grid_shape) / (2 ** downsample_img)
    ty = coords_img[1].reshape(grid_shape) / (2 ** downsample_img)

    dx = tx - grid[1]
    dy = ty - grid[0]

    df = DisplacementField(dx, dy)

    return df


class DatasetDownloader:
    """Class to download an entire dataset.

    Parameters
    ----------
    dataset_id
        Id of the section dataset. Used to determine the 3D matrix.
    downsample_ref
        Downsampling factor of the reference
        space. If set to 1 no downsampling takes place. The reference
        space shape will be divided by `downsample_ref`.
    detection_xy
        Represents the x and y coordinate in the image that will be
        used for determining the slice number in the reference space.
        `p` for coronal slices, `r` for sagittal slices.
    include_expression
        If True then the generator returns 5 objects
        where the last one is the expression image.
    downsample_img
        The downloaded image will have both the height and the width
        downsampled by `2 ** downsample_img`.

    Attributes
    ----------
    metadata
        Needs to be fetched with the `fetch_metadata` method. It contains
        two keys: "dataset" and "images". The values are
        dictionaries storing metadata downloaded from the API.
    """

    def __init__(
        self,
        dataset_id: int,
        downsample_ref: int = 25,
        detection_xy: Tuple[float, float] = (0, 0),
        include_expression: bool = False,
        downsample_img: int = 0,
    ):
        self.dataset_id = dataset_id
        self.downsample_ref = downsample_ref
        self.detection_xy = detection_xy
        self.include_expression = include_expression
        self.downsample_img = downsample_img

        self.metadata: Dict[str, Any] = {}
        # populated by calling `fetch_metadata`

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if "images" not in self.metadata:
            raise RuntimeError("The metadata is empty. Please run `fetch_metadata`")
        return len(self.metadata["images"])

    def fetch_metadata(
        self,
        force_redownload: bool = False,
    ) -> None:
        """Fetch metadata of the dataset.

        This function performs the following steps:
        1. Get metadata for the entire dataset (e.g. `affine_3d`)
        2. Get metadata for all images inside of the dataset (e.g. `affine_2d`)

        Parameters
        ----------
        force_redownload
            If yes, force to redownload the metadata. Otherwise, if
            the metadata have been computed once, they are not computed again.
        """
        if self.metadata and not force_redownload:
            return

        # Prepare query parameters
        parameters_dataset = RMAParameters(
            model="SectionDataSet",
            criteria={
                "id": self.dataset_id,
            },
            include=["alignment3d"],
        )
        parameters_images = RMAParameters(
            model="SectionImage",
            criteria={
                "data_set_id": self.dataset_id,
            },
            include=["alignment2d"],
        )
        # Query the API
        r_datasets = rma_all(parameters_dataset)
        if not r_datasets:
            raise DatasetNotFoundError(
                f"Dataset {self.dataset_id} does not seem to exist"
            )

        r_dataset = r_datasets[0]  # dataset_id is unique
        r_images = rma_all(parameters_images)

        # Extract relevant information
        def extract_template(data, name):
            template = np.array(AFFINE_TEMPLATES[name])
            key = "alignment2d" if name in {"tsv", "tvs"} else "alignment3d"
            return np.vectorize(lambda x: data[key][x])(template)

        dataset_dict = {
            "id": r_dataset["id"],
            "affine_tvr": extract_template(r_dataset, "tvr"),
            "affine_trv": extract_template(r_dataset, "trv"),
            "plane_of_section_id": r_dataset["plane_of_section_id"],
            "section_thickness": r_dataset["section_thickness"],
        }

        images = []
        for r_image in r_images:
            images.append(
                {
                    "id": r_image["id"],
                    "affine_tsv": extract_template(r_image, "tsv"),
                    "affine_tvs": extract_template(r_image, "tvs"),
                    "section_number": r_image["section_number"],
                }
            )

        images.sort(key=lambda x: -int(x["section_number"]))
        metadata = {
            "dataset": dataset_dict,
            "images": images,
        }
        self.metadata = metadata

    def run(
        self,
    ) -> Generator[
        Tuple[int, float, np.ndarray, Optional[np.ndarray], DisplacementField],
        None,
        None,
    ]:
        """Download entire dataset.

        For each image in the dataset, this function performs the following steps:

            1. Query the API to get the `p, i, r` coordinates of the `detection_xy`.
            2. One of the `p, i, r` will become the `slice_coordinate`. For
               coronal datasets it is the `p` and for sagittal ones it is the `r`.
               In other words we assume that the slice is parallel to
               one of the axes.
            3. Use `get_parallel_transform` to get a full mapping between the
               reference space and the image.
            4. Download the image (+ potentially the expression image)
            5. Yield result (order derived from section numbers - highest first)

        Returns
        -------
        res_dict : generator
            Generator yielding consecutive four tuples of
            (image_id, constant_ref_coordinate, img, df).
            The `constant_ref_coordinate` is the dimension in the given axis in microns.
            The `img` is the raw gene expression image with dtype `uint8`.
            The `df` is the displacement field.
            Note that the sorting. If `include_expression=True` then last returned image
            is the processed expression image.
            That is the generator yield (image_id, p, img, df, img_expr).
        """
        if not self.metadata:
            raise RuntimeError("The metadata is empty. Please run `fetch_metadata`")

        metadata_images = self.metadata["images"]
        metadata_dataset = self.metadata["dataset"]

        detection_xy = np.array(self.detection_xy)[:, None]

        plane_of_section = metadata_dataset["plane_of_section_id"]
        if plane_of_section == 1:
            slice_coordinate_ix = 0
            axis = "coronal"
        elif plane_of_section == 2:
            slice_coordinate_ix = 2
            axis = "sagittal"
        else:
            raise ValueError(f"Unrecognized plane of section {plane_of_section}")

        for metadata_image in metadata_images:
            z = metadata_dataset["section_thickness"] * metadata_image["section_number"]
            detection_xy = np.array(
                [
                    [detection_xy[0]],
                    [detection_xy[1]],
                    [z],
                ],
                dtype=np.float32,
            )
            detection_pir = xy_to_pir(
                detection_xy,
                affine_2d=metadata_image["affine_tsv"],
                affine_3d=metadata_dataset["affine_tvr"],
            )
            slice_coordinate = detection_pir[slice_coordinate_ix, 0].item()

            df = get_parallel_transform(
                slice_coordinate,
                affine_2d=metadata_image["affine_tvs"],
                affine_3d=metadata_dataset["affine_trv"],
                downsample_ref=self.downsample_ref,
                axis=axis,
                downsample_img=self.downsample_img,
            )

            image_id = metadata_image["id"]
            img = get_image(
                image_id,
                downsample=self.downsample_img,
            )

            if self.include_expression:
                img_expression = get_image(
                    image_id,
                    expression=True,
                    downsample=self.downsample_img,
                )
            else:
                img_expression = None

            yield image_id, slice_coordinate, img, img_expression, df
