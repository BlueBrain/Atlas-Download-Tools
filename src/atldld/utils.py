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
"""A collection of utilities related to working with Allen's API.

Notes
-----
See the module `atldld.sync.py` for more elaborate functions that use these utils.
Each function here is independent and performs a very specific lower level operation.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import requests

CACHE_FOLDER = os.path.expanduser("~/.atldld/")


def abi_get_request(url):
    """Request url and check the response is valid.

    Parameters
    ----------
    url : str
        URL of the request.

    Returns
    -------
    response : list or dict
         Response to the request.

    Raises
    ------
    ValueError
        If the status_code of the request is different than 200.
        (=Request failed)
    """
    r = requests.get(url)

    if not r.ok:
        raise ValueError("Request failed!")

    response = r.json()["msg"]

    return response


def get_image(image_id, folder=None, expression=False):
    """Get any image from Allen's database just by its id.

    Notes
    -----
    All requested images are stored in the `CACHED_FOLDER` and then read.


    Parameters
    ----------
    image_id : int
        Integer representing an id of the section image.

    folder : str or LocalPath or None
        Local folder where image saved.
        If None then automatically defaults to `CACHE_FOLDER`.

    expression : bool
        If True, retrieve the specified SectionImage's expression mask image.
        Otherwise, retrieve the specified SectionImage.
        See references for details.

    Returns
    -------
    img : np.ndarray
        Downloaded/locally loaded image. The dtype is np.uint8.

    Raises
    ------
    ValueError
        If request to download the image failed.

    References
    ----------
    [1] `AllenSDK API: ImageDownloadApi <https://allensdk.readthedocs.io/
    en/latest/allensdk.api.queries.image_download_api.html#allensdk.api.
    queries.image_download_api.ImageDownloadApi>`_

    """
    folder = folder or CACHE_FOLDER
    folder = str(
        folder
    )  # this should guarantee that also LocalPath works (pytest uses it)

    # Check the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create full path
    additional_specifier = "_expression" if expression else ""
    path = "{}{}{}.jpg".format(folder, image_id, additional_specifier)

    # Check image exists
    if os.path.exists(path):
        img = plt.imread(path)

        if not img.dtype == np.uint8:
            raise ValueError("The dtype needs to be uint8")

        return img

    else:
        image_url = (
            f"https://api.brain-map.org/api/v2/section_image_download/{image_id}"
        )
        if expression:
            image_url += "?view=expression"

        response = requests.get(image_url, stream=True)
        if not response.ok:
            raise ValueError("Request failed!")

        with open(path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return get_image(image_id, expression=expression)


def get_experiment_list_from_gene(gene_name, axis="sagittal"):
    """Get Allen's experiments IDs for a given gene expression name.

    Parameters
    ----------
    gene_name : str
        Gene Expression name.

    axis : {"coronal", "sagittal"}
        Axis of the experiment.

    Returns
    -------
    experiment_list : list
        List of the experiment ID for a given gene expression and a given axis.
    """
    experiment_list = []

    url = (
        "https://api.brain-map.org/api/v2/data/"
        "query.json?criteria=model::SectionDataSet,rma::criteria,[failed$eq'false']"
        f",products[abbreviation$eq'Mouse'],plane_of_section[name$eq'{axis}'],"
        f"genes[acronym$eq'{gene_name}']"
    )

    response = abi_get_request(url)

    for experiment in response:
        experiment_list.append(experiment["id"])

    return experiment_list


def get_2d(image_id, ref2inp=False, add_last=False):
    """For a section image returns the 2D transformation matrix.

    Parameters
    ----------
    image_id : int
        Id of the section image.

    ref2inp : bool, optional
        If True, then reference to input image transformation matrix.
        Otherwise input to reference.

    add_last : bool
        If True then adding a row of [0, 0, 1] to be able to work
        in homogeneous coordinates.

    Returns
    -------
    np.ndarray
        Transformation matrix of shape (2, 3) if add_last=False, else (3, 3).
        The last column represents the translation and
        the potentially 3rd row is just homogeneous coordinates.

    Raises
    ------
    ValueError
        When request not successful.

    """
    # Very quick implementation

    url = (
        f"https://api.brain-map.org/api/v2/data/query.json?"
        f"criteria=model::SectionImage,rma::criteria,"
        f"[id$eq{image_id}],rma::include,alignment2d"
    )

    response = abi_get_request(url)

    temp = response[0]["alignment2d"]
    raw_linear = [
        temp["t{}_0{}".format("vs" if ref2inp else "sv", i)] for i in range(4)
    ]
    raw_translation = [
        temp["t{}_0{}".format("vs" if ref2inp else "sv", i)] for i in [4, 5]
    ]

    a = np.hstack(
        (
            np.reshape(np.array(raw_linear), (2, 2)),
            np.reshape(np.array(raw_translation), (2, 1)),
        )
    )

    if add_last:
        return np.vstack((a, [0, 0, 1]))

    else:
        return a


def get_2d_bulk(dataset_id, ref2inp=False, add_last=False):
    """Get 2D matrices for all images in a given dataset.

    Notes
    -----
    This implementation is significantly faster than simply running
    `get_2d` on each image of the dataset.


    Parameters
    ----------
    dataset_id
        Id of the section dataset.

    ref2inp : bool, optional
        If True, then reference to input image transformation matrix.
        Otherwise input to reference.

    add_last : bool
        If True then adding a row of [0, 0, 1] to each of the 2D matrices
        in order to be able to work in homogeneous coordinates.

    Returns
    -------
    res_dict : dict
        Keys represent image ids and values are tuples of (a, section_number)
        where a is the 2D matrix.

    """
    url = (
        "https://api.brain-map.org/api/v2/data/query.json?"
        "criteria=model::SectionImage,rma::criteria,"
        f"section_data_set[id$eq{dataset_id}],rma::include,alignment2d"
    )
    url += ",rma::options[num_rows$eq2000]"
    # You want to make sure no section images are censored

    response = abi_get_request(url)

    res_dict = {}
    for x in response:
        temp = x["alignment2d"]
        temp_sn = x["section_number"]
        raw_linear = [
            temp["t{}_0{}".format("vs" if ref2inp else "sv", i)] for i in range(4)
        ]
        raw_translation = [
            temp["t{}_0{}".format("vs" if ref2inp else "sv", i)] for i in [4, 5]
        ]

        res_a = np.hstack(
            (
                np.reshape(np.array(raw_linear), (2, 2)),
                np.reshape(np.array(raw_translation), (2, 1)),
            )
        )

        if add_last:
            res_dict[x["id"]] = (np.vstack((res_a, [0, 0, 1])), temp_sn)
        else:
            res_dict[x["id"]] = (res_a, temp_sn)

    return res_dict


def get_3d(dataset_id, ref2inp=False, add_last=False, return_meta=False):
    """For a section dataset returns the 3D transformation matrix.

    Parameters
    ----------
    dataset_id : int
        Id of the section dataset.

    ref2inp : bool, optional
        If True, then reference to input image transformation matrix.
         Otherwise input to reference.

    add_last : bool
        If True then adding a row of [0, 0, 0, 1]
        to be able to work in homogeneous coordinates.

    return_meta : bool
        If True then also the reference_space and the section thickness are returned.

    Returns
    -------
    a : np.ndarray
        Transformation matrix of shape (3, 4) if add_last=False, else (4, 4).
        The last column represents the translation and the potentially 4th row
        is just homogeneous coordinates.

    reference_space : int
        Reference space of the given section dataset.
        Only returned if `return_meta` is True.

    section_thickness : float
        In microns. Only returned if `return_meta` is True.

    Raises
    ------
    ValueError
        When request not successful.

    """
    url = (
        "https://api.brain-map.org/api/v2/data/query.json?"
        "criteria=model::SectionDataSet,rma::criteria,"
        f"[id$eq{dataset_id}],rma::include,alignment3d"
    )

    response = abi_get_request(url)

    temp = response[0]["alignment3d"]
    raw_linear = [
        temp["t{}_{:02d}".format("rv" if ref2inp else "vr", i)] for i in range(9)
    ]
    raw_translation = [
        temp["t{}_{:02d}".format("rv" if ref2inp else "vr", i)] for i in [9, 10, 11]
    ]

    a = np.hstack(
        (
            np.reshape(np.array(raw_linear), (3, 3)),
            np.reshape(np.array(raw_translation), (3, 1)),
        )
    )

    if add_last:
        a = np.vstack((a, [0, 0, 0, 1]))

    # RS
    rs = response[0]["reference_space_id"]

    # thickness
    thickness = response[0]["section_thickness"]

    if return_meta:
        return a, rs, thickness

    else:
        return a


def pir_to_xy_API_single(p, i, r, dataset_id, reference_space=9):
    """Convert an p, i, r in a reference space into a x, y in the image of the dataset.

    Parameters
    ----------
    p : float
        Coronal dimension (anterior -> posterior).
    i : float
        Transversal dimension (superior -> inferior).
        The y (row) coordinate in coronal sections.
    r : float
        Sagittal dimension (left -> right).
        The x (column) coordinate in coronal sections.
    dataset_id : int
        Id of the section dataset.
    reference_space : int, optional
        Reference space for which to perform the computations,
        most likely 9 is the one we always want.

    Returns
    -------
    x : float
        The x coordinate (column) in the image with id `closest_section_image_id`.
    y : float
        The y coordinate (row) in the section image with id `closest_section_image_id`.
    section_number : float
        Section number as calculated by the 3D transformation.
        Since the dataset will never contain exactly this section
        one just uses the closest section image (see `closest_section_imag_id`).

    closest_section_image_id : int
        Id of an image contained in the section dataset such that
        for the given `p`, `i`, `r` input is the closest existing approximation.

    """
    # url = 'http://api.brain-map.org/api/v2/image_to_reference/
    # {}.json?x={}&y={}'.format(image_id, x, y)
    url = (
        "https://api.brain-map.org/api/v2/reference_to_image/"
        f"{reference_space}.json?x={p}&y={i}&z={r}&"
        f"section_data_set_ids={dataset_id}"
    )

    response = abi_get_request(url)

    temp = response[0]["image_sync"]

    x, y = temp["x"], temp["y"]
    section_number = temp["section_number"]
    closest_section_image_id = temp["section_image_id"]

    return x, y, section_number, closest_section_image_id


def xy_to_pir_API_single(x, y, image_id):
    """Convert an x and y in a section image into a p, i, r in the reference space.

    Notes
    -----
    The reference space is always uniquely determined by the
    dataset the image comes from.

    Parameters
    ----------
    x : float
        The x coordinate (column) in the section image with id `image_id`.
    y : float
        The y coordinate (row) in the section image with id `image_id`.
    image_id : int
        Integer representing an id of the section image with id `image_id`.

    Returns
    -------
    p : float
        Coronal dimension (anterior -> posterior).
    i : float
        Transversal dimension (superior -> inferior).
        The y (row) coordinate in coronal sections.
    r : float
        Sagittal dimension (left -> right).
        The x (column) coordinate in coronal sections.
    """
    url = (
        "http://api.brain-map.org/api/v2/image_to_reference/"
        f"{image_id}.json?x={x}&y={y}"
    )

    response = abi_get_request(url)

    temp = response["image_to_reference"]

    p, i, r = temp["x"], temp["y"], temp["z"]

    return p, i, r


class CommonQueries:
    """A collection of very common queries."""

    @staticmethod
    def get_reference_space(dataset_id):
        """Get a reference space id for a given dataset.

        Parameters
        ----------
        dataset_id : int
            Id representing a section dataset.

        Returns
        -------
        reference_space_id : int
            Id representing the reference space.
        """
        url = (
            "http://api.brain-map.org/api/v2/data/query.json?"
            f"criteria=model::SectionDataSet,rma::criteria,[id$eq{dataset_id}]"
        )

        response = abi_get_request(url)

        if not response:
            raise ValueError("No entries for the query (maybe wrong dataset id).")

        reference_space_id = response[0]["reference_space_id"]

        return reference_space_id

    @staticmethod
    def get_axis(dataset_id):
        """Get axis for a given dataset.

        Parameters
        ----------
        dataset_id : int
            Id representing a section dataset.

        Returns
        -------
        axis : str
            Axis of the dataset images. {'sagittal', 'coronal'}
        """
        url = (
            "https://api.brain-map.org/api/v2/data/query.json?"
            f"criteria=model::SectionDataSet,rma::criteria,[id$eq{dataset_id}]"
        )

        response = abi_get_request(url)

        if not response:
            raise ValueError("No entries for the query (maybe wrong dataset id).")

        plane_of_section_id = response[0]["plane_of_section_id"]
        if plane_of_section_id == 1:
            axis = "coronal"
        elif plane_of_section_id == 2:
            axis = "sagittal"
        else:
            raise ValueError(
                f"The plane of section {plane_of_section_id} is not recognized yet."
            )

        return axis
