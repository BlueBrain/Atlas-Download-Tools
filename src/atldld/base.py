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
"""Fundamental building blocks of the project."""

import math
import pathlib

import cv2
import numpy as np
from skimage.transform import AffineTransform, SimilarityTransform


class DisplacementField:
    """
    A class representing a 2D displacement vector field.

    Notes
    -----
    The dtype is enforced to be single-precision (float32) since opencv's remap
    function (used for warping) does not accept double-precision (float64).

    Attributes
    ----------
    delta_x : np.ndarray
        A 2D array of dtype float32 that represents the displacement
        field in the x coordinate (columns). Positive values
        move the pixel to the right, negative move it to the left.
    delta_y : np.ndarray
        A 2D array of dtype float32 that represents the displacement field
        in the y coordinate (rows). Positive values move the pixel down,
        negative pixels move the pixels up.
    """

    def __init__(self, delta_x, delta_y):

        # Checks
        shape_x, shape_y = delta_x.shape, delta_y.shape

        if not len(shape_x) == len(shape_y) == 2:
            raise ValueError("The displacement fields need to be 2D arrays")

        if not shape_x == shape_y:
            raise ValueError(
                "The width and height of x and y displacement field do not match, "
                "{} vs {}".format(shape_x, shape_y)
            )

        self.delta_x = delta_x.astype(np.float32, copy=False)  # for warping
        self.delta_y = delta_y.astype(np.float32, copy=False)

        # Define more attributes
        self.shape = shape_x

    @classmethod
    def from_file(cls, file_path):
        """Load displacement field from a file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to where the file is located.

        Returns
        -------
        DisplacementField
            Instance of the Displacement field.
        """
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        elif isinstance(file_path, pathlib.Path):
            pass

        else:
            raise TypeError(
                "The file path needs to be either a string or a pathlib.Path."
            )

        suffix = file_path.suffix

        if suffix == ".npy":
            deltas_xy = np.load(str(file_path))

            if deltas_xy.ndim != 3:
                raise ValueError("Only supporting 3 dimensional arrays.")

            if deltas_xy.shape[2] != 2:
                raise ValueError(
                    "The last dimensions needs to have 2 elements (delta_x, delta_y)"
                )

            return cls(deltas_xy[..., 0], deltas_xy[..., 1])

        else:
            raise ValueError("Unsupported suffix {}".format(suffix))

    @classmethod
    def from_transform(cls, f_x, f_y):
        """Instantiate displacement field from actual transformations.

        Parameters
        ----------
        f_x : np.array
            2D array of shape (h, w) representing the x coordinate
            of the transformation.
        f_y : np.array
            2D array of shape (h, w) representing the y coordinate
            of the transformation.

        Returns
        -------
        DisplacementField
            Instance of the Displacement field.
        """
        # checks
        shape_x, shape_y = f_x.shape, f_y.shape

        if not len(shape_x) == len(shape_y) == 2:
            raise ValueError("The transforms need to be 2D arrays")

        if not shape_x == shape_y:
            raise ValueError(
                "The width and height of x and y transforms do not match, "
                "{} vs {}".format(shape_x, shape_y)
            )

        shape = shape_x

        y, x = np.indices(shape, dtype=np.float32)

        delta_x = f_x - x
        delta_y = f_y - y

        return DisplacementField(delta_x, delta_y)

    def __eq__(self, other):
        """Equality."""
        if not isinstance(other, DisplacementField):
            raise TypeError(
                "The right hand side object is not DisplacementField but {}".format(
                    type(other)
                )
            )

        return np.allclose(self.delta_x, other.delta_x) and np.allclose(
            self.delta_y, other.delta_y
        )

    def __mul__(self, c):
        """Multiplication by a constant from the right.

        Parameters
        ----------
        c : int or float
            A number.

        Returns
        -------
        result : DisplacementField
            An instance of DisplacementField where both the `delta_x' and `delta_y`
            were elementwise multiplied by `c`.

        Raises
        ------
        TypeError
            If `c` is not int or float.
        """
        if not isinstance(c, (int, float)):
            raise TypeError("The constant c needs to be a number.")

        return DisplacementField(delta_x=c * self.delta_x, delta_y=c * self.delta_y)

    def __rmul__(self, c):
        """Multiplication by a constant from the left.

        Notes
        -----
        Since we want this to be commutative we simply delegate
        all the logic to `__mul__` method.

        Parameters
        ----------
        c : int or float
            A number.

        Returns
        -------
        result : DisplacementField
            An instance of DisplacementField where both the `delta_x' and `delta_y`
            were elementwise multiplied by `c`.

        Raises
        ------
        TypeError
            If `c` is not int or float.
        """
        return self * c

    @property
    def norm(self):
        """Norm for each pixel."""
        return np.sqrt(np.square(self.delta_x) + np.square(self.delta_y))

    @property
    def transformation(self):
        """Output the actual transformation rather than the displacement field.

        Returns
        -------
        f_x : np.ndarray
            A 2D array of dtype float32. For each pixel in the fixed image what
            is the corresponding x coordinate in the moving image.
        f_y : np.ndarray
            A 2D array of dtype float32. For each pixel in the fixed image what
            is the corresponding y coordinate in the moving image.
        """
        x, y = np.meshgrid(
            np.arange(self.shape[1], dtype=np.float32),
            np.arange(self.shape[0], dtype=np.float32),
            copy=False,
        )  # will guarantee the output is float32

        f_x = x + self.delta_x
        f_y = y + self.delta_y

        return f_x, f_y

    def save(self, path):
        """Save displacement field as a .npy file.

        Notes
        -----
        Can be loaded via `DisplacementField.from_file` class method.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the file. Needs to end with `.npy`.
        """
        path = pathlib.Path(path)

        if path.suffix == ".npy":
            pass

        else:
            raise ValueError("Invalid suffix {}".format(path.suffix))

        np.save(path, np.stack([self.delta_x, self.delta_y], axis=2))

    def warp(self, img, interpolation="linear", border_mode="constant", c=0):
        """Warp an input image based on the inner displacement field.

        Parameters
        ----------
        img : np.ndarray
            Input image to which we will apply the transformation. Note
            that certain dtypes (e.g. `np.float64`) are not supported
            by OpenCV.

        interpolation : str, {'nearest', 'linear', 'cubic', 'area', 'lanczos'}
            Regular grid interpolation method to be used.
        border_mode : str, {'constant', 'replicate', 'reflect',
        'wrap', 'reflect101', 'transparent'}
            How to fill outside of the range values.
            See references for detailed explanation.
        c : float
            Only used if `border_mode='constant'` and represents the fill value.

        Returns
        -------
        warped_img : np.ndarray
            Warped image.
        """
        interpolation_mapper = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        border_mode_mapper = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "wrap": cv2.BORDER_WRAP,
            "reflect_101": cv2.BORDER_REFLECT101,
            "transparent": cv2.BORDER_TRANSPARENT,
        }

        if interpolation not in interpolation_mapper:
            raise KeyError(
                "Unsupported interpolation, available options: {}".format(
                    interpolation_mapper.keys()
                )
            )

        if border_mode not in border_mode_mapper:
            raise KeyError(
                "Unsupported border_mode, available options: {}".format(
                    border_mode_mapper.keys()
                )
            )

        fx, fy = self.transformation

        return cv2.remap(
            img,
            fx,
            fy,
            interpolation=interpolation_mapper[interpolation],
            borderMode=border_mode_mapper[border_mode],
            borderValue=c,
        )


def affine(shape, matrix):
    """Affine transformation encoded in a 2 x 3 matrix.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).
    matrix : np.ndarray
        Transformation matrix of the shape 2 x 3.

    Raises
    ------
    ValueError
        In case the transformation matrix has a wrong shape.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.
    delta_y : np.ndarray
        Displacement vector field of the y coordinates.
    """
    if matrix.shape != (3, 3):
        raise ValueError(
            "The shape of transformation matrix is {}, correct is (3, 3).".format(
                matrix.shape
            )
        )

    tform = AffineTransform(matrix)

    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).astype(float)

    coords_after = tform(coords)
    coords_delta = coords_after - coords

    delta_x = np.reshape(coords_delta[:, 0], shape)
    delta_y = np.reshape(coords_delta[:, 1], shape)

    return delta_x, delta_y


def affine_simple(
    shape,
    scale_x=1,
    scale_y=1,
    rotation=0,
    translation_x=0,
    translation_y=0,
    shear=0,
    apply_centering=True,
):
    """Just a human version of affine mapping.

    Notes
    -----
    Instead of specifying the whole matrix one can just specify
    all the understandable quantities.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).
    scale_x : float
        Scale on the x axis. If scale_x < 1 then zoom out, if scale_x > 1 zoom in.
    scale_y : float
        Scale on the y axis. If scale_y < 1 then zoom out, if scale_y > 1 zoom in.
    rotation : float
          Rotation angle in counter-clockwise direction as radians.
    translation_x : float
        Translation in the x direction.
        If translation_x > 0 then to the right, else to the left.
    translation_y : float
        Translation in the y direction. If translation_y > 0 then down, else to the up.
    shear : float
        Shear angle in counter-clockwise direction as radians.
    apply_centering : bool
        If True then (h // 2 - 0.5, w // 2 - 0.5) is considered a center of the image.
        And before performing all the other operations the image is first
        shifted so that the center corresponds to (0, 0). Then the actual
        transformation is applied and after that the image is shifted
        into the original center.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.
    delta_y : np.ndarray
        Displacement vector field of the y coordinates.
    """
    matrix = np.array(
        [
            [
                scale_x * math.cos(rotation),
                -scale_y * math.sin(rotation + shear),
                translation_x,
            ],
            [
                scale_x * math.sin(rotation),
                scale_y * math.cos(rotation + shear),
                translation_y,
            ],
            [0, 0, 1],
        ]
    )

    if not apply_centering:
        return affine(shape, matrix)

    center_rc = np.array([(shape[0] / 2) - 0.5, (shape[1] / 2) - 0.5])
    center_xy = np.array([center_rc[1], center_rc[0]])

    tform1 = SimilarityTransform(translation=center_xy)
    tform2 = SimilarityTransform(matrix=matrix)
    tform3 = SimilarityTransform(translation=-center_xy)
    tform = tform3 + tform2 + tform1

    return affine(shape, tform.params)
