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

from typing import Generator, Tuple, Union

import numpy as np

from atldld.base import DisplacementField
from atldld.utils import (
    CommonQueries,
    get_2d_bulk,
    get_3d,
    get_image,
    xy_to_pir_API_single,
)


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
        `tvr_**` entries.
    affine_3d
        Matrix of shape `(3, 4)` representing a 3D affine transformation. It
        can be retrieved from the dataset metadata via the Allen Brain API.
        More specifically, it is stored under the `tsv_**` entries.

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


def download_parallel_dataset(
    dataset_id: int,
    downsample_ref: int = 25,
    detection_xy: Tuple[float, float] = (0, 0),
    include_expression: bool = False,
    downsample_img: int = 0,
) -> Generator[
    Union[
        Tuple[int, float, np.ndarray, DisplacementField],
        Tuple[int, float, np.ndarray, DisplacementField, np.ndarray],
    ],
    None,
    None,
]:
    """Download entire dataset.

    This function performs the following steps:

    1. Get metadata for the entire dataset (e.g. `affine_3d`)
    2. Get metadata for all images inside of the dataset (e.g. `affine_2d`)
    3. For each image in the dataset do the following

        a. Query the API to get the `p, i, r` coordinates of the `detection_xy`.
        b. One of the `p, i, r` will become the `slice_coordinate`. For
           coronal datasets it is the `p` and for sagittal ones it is the `r`.
           In other words we assume that the slice is parallel to
           one of the axes.
        c. Use `get_parallel_transform` to get a full mapping between the
           reference space and the image.
        d. Download the image (+ potentially the expression image)
        e. Yield result (order derived from section numbers - highest first)

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
    metadata_2d_dict = get_2d_bulk(
        dataset_id,
        ref2inp=True,
    )
    metadata_2d = sorted(
        [
            (image_id, affine_2d, section_number)
            for image_id, (affine_2d, section_number) in metadata_2d_dict.items()
        ],
        key=lambda x: -int(x[2]),  # we use section_number for sorting
    )

    affine_3d = get_3d(
        dataset_id,
        ref2inp=True,
        return_meta=False,
    )
    axis = CommonQueries.get_axis(dataset_id)

    for image_id, affine_2d, _ in metadata_2d:
        p, i, r = xy_to_pir_API_single(*detection_xy, image_id=image_id)
        slice_ref_coordinate = p if axis == "coronal" else r

        df = get_parallel_transform(
            slice_ref_coordinate,
            affine_2d,
            affine_3d,
            downsample_ref=downsample_ref,
            axis=axis,
            downsample_img=downsample_img,
        )

        img = get_image(image_id, downsample=downsample_img)

        if not include_expression:
            yield image_id, slice_ref_coordinate, img, df
        else:
            img_expression = get_image(
                image_id,
                expression=True,
                downsample=downsample_img,
            )
            yield image_id, slice_ref_coordinate, img, df, img_expression
