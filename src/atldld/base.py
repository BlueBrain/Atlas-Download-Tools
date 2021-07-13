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
"""Fundamental building blocks of the project.

Notes
-----
This module does not import any other module except for zoo.
Be careful to keep this logic in order to prevent cyclical imports.
"""

import math
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from skimage.transform import AffineTransform, SimilarityTransform, resize

GLOBAL_CACHE_FOLDER = pathlib.Path.home() / ".atldld"


def affine(shape, matrix=None):
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
    if matrix is None:
        matrix = np.eye(3)

    if matrix.shape == (2, 3):
        matrix = np.vstack(
            (matrix, [0, 0, 1])
        )  # just add the homogeneous coordinates parts

    if matrix.shape != (3, 3):
        raise ValueError(
            "The shape of affine transformation matrix is {}, correct is (3, 3)".format(
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

    @classmethod
    def generate(cls, shape, approach="identity", **kwargs):
        """Construct different displacement vector fields (DVF) via factory method.

        Parameters
        ----------
        shape : tuple
            A tuple representing the (height, width) of the displacement field.
            Note that if multiple channels passed then only the height
            and width is extracted.
        approach : str, {'affine', 'affine_simple', 'identity'}
            What approach to use for generating the DVF.
        kwargs : dict
            Additional parameters that are passed into the the given approach function.

        Returns
        -------
        DisplacementField
            An instance of a Displacement field.
        """
        # Check - extremely important since no checks in the zoo
        if len(shape) == 3:
            shape_ = shape[
                :2
            ]  # to make it easier for the user who passes img.shape of an RGB image

        elif len(shape) == 2:
            shape_ = shape

        else:
            raise ValueError(
                "The length of shape needs to be either 2 or 3, {} given".format(
                    len(shape)
                )
            )

        if approach == "affine":
            kw = ["matrix"]
            kwargs_affine = {
                k: v for k, v in kwargs.items() if k in kw
            }  # Check if passed any
            delta_x, delta_y = affine(shape_, **kwargs_affine)

        elif approach == "affine_simple":
            kw = [
                "scale_x",
                "scale_y",
                "rotation",
                "translation_x",
                "translation_y",
                "shear",
                "apply_centering",
            ]
            kwargs_affine_simple = {
                k: v for k, v in kwargs.items() if k in kw
            }  # Check if passed any
            delta_x, delta_y = affine_simple(shape_, **kwargs_affine_simple)

        elif approach == "identity":
            kw = []
            delta_x, delta_y = np.zeros(shape_), np.zeros(shape_)

        else:
            raise ValueError("The approach {} is not valid".format(approach))

        # Check if no illegal arguments (now its too late but beter than never:D)
        allowed_kw = set(kw)
        passed_kw = set(kwargs)

        if not passed_kw.issubset(allowed_kw):
            diff = passed_kw - allowed_kw
            raise ValueError(
                "Illegal arguments passed for approach {}: {}".format(approach, diff)
            )

        return cls(delta_x=delta_x, delta_y=delta_y)

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

        x, y = np.meshgrid(list(range(shape[1])), list(range(shape[0])))

        delta_x = f_x - x
        delta_y = f_y - y

        return DisplacementField(delta_x, delta_y)

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

        self.delta_x = delta_x.astype(np.float32)
        self.delta_y = delta_y.astype(np.float32)

        # Define more attributes
        self.shape = shape_x

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

    def __call__(self, other, interpolation="linear", border_mode="replicate", c=0):
        """Composition of transformations.

        Notes
        -----
        This composition is only approximate since we need to approximate `self`
        on off-grid elements. Negative side effect is that composing with inverse
        will not necessarily lead to identity.

        Parameters
        ----------
        other : DisplacementField
            An inner DVF.
        interpolation : str, {'nearest', 'linear', 'cubic', 'area', 'lanczos'}
            Regular grid interpolation method to be used.
        border_mode : str,
        {'constant', 'replicate', 'reflect', 'wrap', 'reflect101', 'transparent'}
            How to fill outside of the range values.
            See references for detailed explanation.
        c : float
            Only used if `border_mode='constant'` and represents the fill value.

        Returns
        -------
        composition : DisplacementField
            Let F: x -> x + self and G: x -> x + other.
            Then the composition represents x: F(G(x)) - x.
        """
        if not isinstance(other, DisplacementField):
            raise TypeError(
                "The inner object is not DisplacementField but {}".format(type(other))
            )

        if self.shape != other.shape:
            raise ValueError("Cannot compose DVF of different shapes!")

        # Think about self as 2 images delta_x and delta_y,
        # and the final transformation also as 2 images with
        # intentieties being equal to the output vector.

        x, y = np.meshgrid(list(range(self.shape[1])), list(range(self.shape[0])))
        delta_x = (
            other.warp(
                x + self.delta_x,
                interpolation=interpolation,
                border_mode=border_mode,
                c=c,
            )
            - x
        )
        delta_y = (
            other.warp(
                y + self.delta_y,
                interpolation=interpolation,
                border_mode=border_mode,
                c=c,
            )
            - y
        )

        return DisplacementField(delta_x, delta_y)

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
    def average_displacement(self):
        """Average displacement per pixel."""
        return self.norm.mean()

    @property
    def delta_x_scaled(self):
        """Scaled version of delta_x."""
        return self.delta_x / self.shape[1]

    @property
    def delta_y_scaled(self):
        """Scaled version of delta_y."""
        return self.delta_y / self.shape[0]

    @property
    def is_valid(self):
        """Check whether both delta_x and delta_y finite."""
        return np.all(np.isfinite(self.delta_x)) and np.all(np.isfinite(self.delta_y))

    @property
    def jacobian(self):
        """Compute determinant of a Jacobian per each pixel."""
        delta_x = self.delta_x
        delta_y = self.delta_y

        a_11 = np.zeros(self.shape)
        a_12 = np.zeros(self.shape)
        a_21 = np.zeros(self.shape)
        a_22 = np.zeros(self.shape)

        # inside (symmetric)
        a_11[:, 1:-1] = 1 + (-delta_x[:, :-2] + delta_x[:, 2:]) / 2
        a_12[1:-1, :] = (-delta_x[:-2, :] + delta_x[2:, :]) / 2
        a_21[:, 1:-1] = (-delta_y[:, :-2] + delta_y[:, 2:]) / 2
        a_22[1:-1, :] = 1 + (-delta_y[:-2, :] + delta_y[2:, :]) / 2

        # edges (one-sided)
        a_11[:, 0] = 1 + (delta_x[:, 1] - delta_x[:, 0])
        a_11[:, -1] = 1 + (delta_x[:, -1] - delta_x[:, -2])

        a_12[0, :] = delta_x[1, :] - delta_x[0]
        a_12[-1, :] = delta_x[-1, :] - delta_x[-2, :]

        a_21[:, 0] = delta_y[:, 1] - delta_y[:, 0]
        a_21[:, -1] = delta_y[:, -1] - delta_y[:, -2]

        a_22[0, :] = 1 + delta_y[1, :] - delta_y[0]
        a_22[-1, :] = 1 + delta_y[-1, :] - delta_y[-2, :]

        res = np.multiply(a_11, a_22) - np.multiply(a_12, a_21)

        return res

    @property
    def n_pixels(self):
        """Count the number of pixels in the displacement field.

        Notes
        -----
        Number of channels is ignored.
        """
        return np.prod(self.shape[:2])

    @property
    def norm(self):
        """Norm for each pixel."""
        return np.sqrt(np.square(self.delta_x) + np.square(self.delta_y))

    @property
    def outsiders(self):
        """For each pixels determines whether it is mapped outside of the image.

        Notes
        -----
        An important thing to look out for since for each outsider
        the interpolator cannot use grid interpolation.
        """
        x, y = np.meshgrid(list(range(self.shape[1])), list(range(self.shape[0])))
        f_x, f_y = x + self.delta_x, y + self.delta_y

        return np.logical_or(
            np.logical_or(0 > f_x, f_x >= self.shape[1]),
            np.logical_or(0 > f_y, f_y >= self.shape[0]),
        )

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

    def adjust(self, delta_x_max=None, delta_y_max=None, force_inside_border=True):
        """Adjust the displacement field.

        Notes
        -----
        Not in place, returns a modified instance.

        Parameters
        ----------
        delta_x_max : float
            Maximum absolute size of delta_x. If None, no limit is imposed.
        delta_y_max : float
            Maximum absolute size of delta_y. If None, no limit is imposed.
        force_inside_border : bool
            If True, then all displacement vector that would result
            in leaving the image are clipped.

        Returns
        -------
        DisplacementField
           Adjusted DisplacementField.
        """
        eps_final = 1e-8  # just to make sure everything is within the image region

        sign_x = np.sign(self.delta_x)
        sign_y = np.sign(self.delta_y)

        new_delta_x = (
            np.minimum(delta_x_max, abs(self.delta_x)) * sign_x
            if delta_x_max is not None
            else self.delta_x
        )
        new_delta_y = (
            np.minimum(delta_y_max, abs(self.delta_y)) * sign_y
            if delta_y_max is not None
            else self.delta_y
        )

        if force_inside_border:
            # will result in a lot of pixels mapping to the border
            c_matrix_x = np.ones(
                self.shape
            )  # minimum scaling that results in x not being outside
            c_matrix_y = np.ones(
                self.shape
            )  # minimum scaling that results in y not being outside

            # Preparation
            x, y = np.meshgrid(list(range(self.shape[1])), list(range(self.shape[0])))
            max_u = y
            max_d = self.shape[0] - y - 1
            max_r = self.shape[1] - x - 1
            max_l = x

            # Max delta_x - depends whether positive or negative
            max_delta_x = (
                np.ones(self.shape) * np.inf
            )  # by default it can be as large as you want
            ix_delta_x_pos = self.delta_x > 0
            ix_delta_x_zero = self.delta_x == 0
            ix_delta_x_neg = self.delta_x < 0

            max_delta_x[ix_delta_x_pos] = max_r[ix_delta_x_pos]
            max_delta_x[ix_delta_x_neg] = max_l[ix_delta_x_neg]

            # Max delta_y - depends whether positive or negative
            max_delta_y = np.ones(self.shape) * np.inf
            ix_delta_y_pos = self.delta_y > 0
            ix_delta_y_zero = self.delta_y == 0
            ix_delta_y_neg = self.delta_y < 0

            # Experimental
            max_delta_y[ix_delta_y_pos] = max_d[ix_delta_y_pos]
            max_delta_y[ix_delta_y_neg] = max_u[ix_delta_y_neg]

            c_matrix_x[~ix_delta_x_zero] = np.minimum(
                1, max_delta_x[~ix_delta_x_zero] / abs(new_delta_x[~ix_delta_x_zero])
            )
            c_matrix_y[~ix_delta_y_zero] = np.minimum(
                1, max_delta_y[~ix_delta_y_zero] / abs(new_delta_y[~ix_delta_y_zero])
            )

            c_matrix = np.minimum(c_matrix_x, c_matrix_y) * (1 - eps_final)

            new_delta_x = c_matrix * new_delta_x
            new_delta_y = c_matrix * new_delta_y

        return DisplacementField(new_delta_x, new_delta_y)

    def mask(self, mask_matrix, fill_value=0):
        """Mask a displacement field.

        Notes
        -----
        Not in place, returns a modified instance.

        Parameters
        ----------
        mask_matrix : np.array
            An array of dtype=bool where True represents a pixel that is supposed
            to be unchanged. False pixels are filled with `fill_value`.
        fill_value : float or tuple
            Value to fill the False pixels with.
            If tuple then fill_value_x, fill_value_y

        Returns
        -------
        DisplacementField
            A new DisplacementField instance accordingly masked.
        """
        if not mask_matrix.shape == self.shape:
            raise ValueError(
                "The mask array has an incorrect shape of {}.".format(mask_matrix.shape)
            )

        if not mask_matrix.dtype == bool:
            raise TypeError(
                "The dtype of the array needs to be a bool, current dtype {}".format(
                    mask_matrix.dtype
                )
            )

        delta_x_masked = self.delta_x.copy()
        delta_y_masked = self.delta_y.copy()

        if isinstance(fill_value, tuple):
            fill_value_x, fill_value_y = fill_value

        elif isinstance(fill_value, (float, int)):
            fill_value_x, fill_value_y = fill_value, fill_value

        else:
            raise TypeError("Incorrect type {} of fill_value".format(type(fill_value)))

        delta_x_masked[~mask_matrix] = fill_value_x
        delta_y_masked[~mask_matrix] = fill_value_y

        return DisplacementField(delta_x_masked, delta_y_masked)

    def plot_dvf(self, ds_f=8, figsize=(15, 15), ax=None):
        """Plot displacement vector field.

        Notes
        -----
        Still works in a weird way.

        Parameters
        ----------
        ds_f : int
            Downsampling factor,
            i.e if `ds_f=8` every 8-th row and every 8th column printed.
        figsize : tuple
            Size of the figure.
        ax : matplotlib.Axes
            Axes upon which to plot. If None, create a new one

        Returns
        -------
        ax : matplotlib.Axes
            Axes with the visualization.
        """
        x, y = np.meshgrid(list(range(self.shape[1])), list(range(self.shape[0])))

        if ax is None:
            _, ax_quiver = plt.subplots(figsize=figsize)

        else:
            ax_quiver = ax

        ax_quiver.invert_yaxis()
        ax_quiver.quiver(
            x[::ds_f, ::ds_f],
            y[::ds_f, ::ds_f],
            self.delta_x[::ds_f, ::ds_f],
            -self.delta_y[::ds_f, ::ds_f],
        )  # matplotlib has positive delta y as up, in our case its down

        return ax_quiver

    def plot_outside(self, figsize=(15, 15), ax=None):
        """Plot all pixels that are mapped outside of the image.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        ax : matplotlib.Axes
            Axes upon which to plot. If None, create a new one

        Returns
        -------
        ax : matplotlib.Axes
            Axes with the visualization.
        """
        res = np.zeros(self.shape, dtype=float)

        if ax is None:
            _, ax_outside = plt.subplots(figsize=figsize)

        else:
            ax_outside = ax

        res[self.outsiders] = 1

        ax_outside.imshow(res, cmap="gray")

        return ax_outside

    def plot_ranges(
        self, freq=10, figsize=(15, 10), kwargs_domain=None, kwargs_range=None, ax=None
    ):
        """Plot domain and the range of the mapping.

        Parameters
        ----------
        freq : int
            Take every `freq`th pixel. The higher the more sparse.
        figsize : tuple
            Size of the figure.
        kwargs_domain : dict or None
            If ``dict`` then matplotlib kwargs to be passed into the domain scatter.
        kwargs_range : dict or None
            If ``dict`` then matplotlib kwargs to be passed into the range scatter.
        ax : matplotlib.Axes
            Axes upon which to plot. If None, create a new one.

        Returns
        -------
        ax : matplotlib.Axes
            Axes with the visualization.
        """
        # original range
        h, w = self.shape
        tx, ty = self.transformation

        x, y = [], []
        x_r, y_r = [], []

        i = 0
        for r in range(h):
            for c in range(w):
                i += 1
                if i % freq == 0:
                    x.append(c)
                    y.append(h - r)
                    x_r.append(tx[r, c])
                    y_r.append(h - ty[r, c])

        kwargs_domain = kwargs_domain or {"s": 0.1, "color": "blue"}
        kwargs_range = kwargs_range or {"s": 0.1, "color": "green"}

        if ax is None:
            _, ax_ranges = plt.subplots(figsize=figsize)
        else:
            ax_ranges = ax

        ax_ranges.scatter(x, y, label="Domain", **kwargs_domain)
        ax_ranges.scatter(x_r, y_r, label="Range", **kwargs_range)
        ax_ranges.legend()

        return ax_ranges

    def resize(self, new_shape):
        """Calculate a resized displacement vector field.

        Goal: df_resized.warp(img) ~ resized(df.warp(img))

        Parameters
        ----------
        new_shape : tuple
            Represents (new_height, new_width) of the resized displacement field.

        Returns
        -------
        DisplacementField
            New DisplacementField with a shape of new_shape.
        """
        if not isinstance(new_shape, tuple):
            raise TypeError("Incorrect type of new_shape: {}".format(type(new_shape)))

        if not len(new_shape) == 2:
            raise ValueError("The length of new shape must be 2")

        f_x, f_y = self.transformation
        new_f_x = resize(f_x, output_shape=new_shape)
        new_f_y = resize(f_y, output_shape=new_shape)

        return DisplacementField.from_transform(new_f_x, new_f_y)

    def resize_constant(self, new_shape):
        """Calculate the resized displacement vector field.

        That will have the same effect on original image.
        Goal:  upsampled(df.warp(img_downsampled)) ~  df_resized.warp(img).

        Parameters
        ----------
        new_shape : tuple
            Represents (new_height, new_width) of the resized displacement field.

        Returns
        -------
        DisplacementField
            New DisplacementField with a shape of new_shape.

        Notes
        -----
        Very useful when we perform registration on a smaller resolution image
        and then we want to resize it back to the original higher resolution shape.
        """
        fx, fy = self.transformation

        x_ratio, y_ratio = new_shape[1] / self.shape[1], new_shape[0] / self.shape[0]

        fx_, fy_ = fx * x_ratio, fy * y_ratio

        new_f_x = resize(fx_, output_shape=new_shape)
        new_f_y = resize(fy_, output_shape=new_shape)

        return DisplacementField.from_transform(new_f_x, new_f_y)

    def save(self, path):
        """Save displacement field as a .npy file.

        Notes
        -----
        Can be loaded via `DisplacementField.from_file` class method.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the file.
        """
        path = pathlib.Path(path)

        if path.suffix == "":
            path = path.with_suffix(".npy")

        elif path.suffix == ".npy":
            pass

        else:
            raise ValueError("Invalid suffix {}".format(path.suffix))

        np.save(path, np.stack([self.delta_x, self.delta_y], axis=2))

    def warp(self, img, interpolation="linear", border_mode="constant", c=0):
        """Warp an input image based on the inner displacement field.

        Parameters
        ----------
        img : np.ndarray
            Input image to which we will apply the transformation.
            Currently the only 3 supported dtypes are uint8, float32 and float64.
            The logic is for the `warped_img`  to have the dtype
            (input dtype - output dtype).
                * uint8 - uint8
                * float32 - float32
                * float64 - float32
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
            Warped image. Note that the dtype will be the same as the input `img`.
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

        dtype = img.dtype

        if dtype == np.float32 or dtype == np.uint8:
            img_ = img

        elif dtype == np.float64:
            img_ = img.astype(np.float32)
            dtype = np.float32

        else:
            raise ValueError("Unsupported dtype {}.".format(dtype))

        fx, fy = self.transformation

        return cv2.remap(
            img_,
            fx,
            fy,
            interpolation=interpolation_mapper[interpolation],
            borderMode=border_mode_mapper[border_mode],
            borderValue=c,
        )

    def warp_annotation(self, img, approach="opencv"):
        """Warp an input annotation image based on the displacement field.

        If displacement falls outside of the image the logic is to replicate the border.
        This approach guarantees that no new labels are created.

        Notes
        -----
        If approach is 'scipy' then calls ``scipy.spatial.cKDTree``
        in the background with default Euclidian distance
        and exactly 1 nearest neighbor.

        Parameters
        ----------
        img : np.ndarray
            Input annotation image. The allowed dtypes are currently int8, int16, int32
        approach : str, {'scipy', 'opencv'}
            Approach to be used. Currently 'opencv' way faster.

        Returns
        -------
        warped_img : np.ndarray
            Warped image.
        """
        allowed_dtypes = ["int8", "int16", "int32"]
        input_dtype = img.dtype

        # CHECKS
        if not any([input_dtype == x for x in allowed_dtypes]):
            raise ValueError("The only allowed dtypes are {}".format(allowed_dtypes))

        if approach == "scipy":
            x, y = np.meshgrid(list(range(self.shape[1])), list(range(self.shape[0])))

            temp_all = np.hstack(
                (y.reshape(-1, 1), x.reshape(-1, 1), img[y, x].reshape(-1, 1))
            )

            inst = NearestNDInterpolator(temp_all[:, :2], temp_all[:, 2])
            x_r, y_r = x.ravel(), y.ravel()

            coords = np.hstack(
                (
                    (y_r + self.delta_y.ravel()).reshape(-1, 1),
                    (x_r + self.delta_x.ravel()).reshape(-1, 1),
                )
            )

            return inst(coords).reshape(self.shape).astype(input_dtype)

        elif approach == "opencv":
            # opencv keeps the same dtype apparently
            fx, fy = self.transformation

            return cv2.remap(
                img, fx, fy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
            )

        else:
            raise ValueError("Unrecognized approach {}".format(approach))
