"""A collection of utilities related to working with Allen's API.

Notes
-----
See the module `atlutils.sync.py` for more elaborate functions that use these utils.
Each function here is independent and performs a very specific lower level operation.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from allensdk.api.queries.image_download_api import ImageDownloadApi

CACHE_FOLDER = os.path.expanduser("~/.atlutils/")


def get_image(image_id, folder=None, **kwargs):
    """Get any image from Allen's database just by its id.

    Notes
    -----
    All requested images are stored in the `CACHED_FOLDER` and then read.


    Parameters
    ----------
    image_id : int
        Integer representing an id of the section image.

    folder : str or LocalPath or None
        Local folder where image saved. If None then automatically defaults to `CACHE_FOLDER`.

    **kwargs
        Additional parameters to be passed onto the `download_image` method of ``ImageDownloadApi``. See
        See references for details.

    Returns
    -------
    img : np.ndarray
        Downloaded/locally loaded image. The dtype is np.uint8.

    References
    ----------
    [1] https://allensdk.readthedocs.io/en/latest/allensdk.api.queries.image_download_api.html#allensdk.api.queries.image_download_api.ImageDownloadApi  # noqa

    """
    folder = folder or CACHE_FOLDER
    folder = str(
        folder
    )  # this should guarantee that also LocalPath works (pytest uses it)

    # Check the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create full path
    additional_speficier = "_".join(
        sorted(["{}_{}".format(k, v) for k, v in kwargs.items()])
    )
    if additional_speficier:
        additional_speficier = "_{}".format(additional_speficier)
    path = "{}{}{}.jpg".format(folder, image_id, additional_speficier)

    # Check image exists
    if os.path.exists(path):
        img = plt.imread(path)

        if not img.dtype == np.uint8:
            raise ValueError("The dtype needs to be uint8")

        return img

    else:

        img_api = ImageDownloadApi()
        img_api.download_image(image_id, file_path=path, **kwargs)
        return get_image(image_id, **kwargs)


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

    url = "http://api.brain-map.org/api/v2/data/query.json?"
    url += (
        "criteria=model::SectionImage,rma::criteria,"
        "[id$eq{}],rma::include,alignment2d".format(image_id)
    )

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Request failed!")

    temp = r.json()["msg"][0]["alignment2d"]
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
    url = "http://api.brain-map.org/api/v2/data/query.json?"
    url += (
        "criteria=model::SectionImage,rma::criteria,"
        "section_data_set[id$eq{}],rma::include,alignment2d".format(dataset_id)
    )
    url += ",rma::options[num_rows$eq2000]"
    # You want to make sure no section images are censored

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Request failed!")

    res_dict = {}
    for x in r.json()["msg"]:
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
    url = "http://api.brain-map.org/api/v2/data/query.json?"
    url += (
        "criteria=model::SectionDataSet,rma::criteria,"
        "[id$eq{}],rma::include,alignment3d".format(dataset_id)
    )
    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Request failed!")

    temp = r.json()["msg"][0]["alignment3d"]
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
    rs = r.json()["msg"][0]["reference_space_id"]

    # thickness
    thickness = r.json()["msg"][0]["section_thickness"]

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
    url = "http://api.brain-map.org/api/v2/reference_to_image/"
    url += "{}.json?x={}&y={}&z={}&section_data_set_ids={}".format(
        reference_space, p, i, r, dataset_id
    )

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Request failed!")

    temp = r.json()["msg"][0]["image_sync"]

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
    url = "http://api.brain-map.org/api/v2/image_to_reference/{}.json?x={}&y={}".format(
        image_id, x, y
    )

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Request failed!")

    temp = r.json()["msg"]["image_to_reference"]

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
        url = "http://api.brain-map.org/api/v2/data/query.json?"
        url += "criteria=model::SectionDataSet,rma::criteria,[id$eq{}]".format(
            dataset_id
        )
        r = requests.get(url)

        if r.status_code != 200:
            raise ValueError("Request failed!")

        raw = r.json()["msg"]
        if not raw:
            raise ValueError("No entries for the query (maybe wrong dataset id).")

        reference_space_id = raw[0]["reference_space_id"]

        return reference_space_id
