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
"""Test for base.py module."""

import pathlib
from typing import Union

import numpy as np
import pytest

from atldld.base import DisplacementField, affine, affine_simple

SUPPORTED_DTYPES_ANNOTATIONS = ["int8", "int16", "int32"]
WARP_INTERPOLATIONS = ["linear"]
WARP_BORDER_MODES = ["constant", "replicate"]


class TestDisplacementFieldEssentials:
    """A collection of tests that focus on the
    fundamental functioning of the DisplacementField class.

    Notes
    -----
    Mostly focused on the properties.
    """

    def test_wrong_construction(self):
        with pytest.raises(ValueError, match="2D"):
            delta = np.ones((2, 2, 2))  # 3D not supported
            DisplacementField(delta, delta)

        with pytest.raises(ValueError, match="do not match"):
            DisplacementField(np.ones((2, 3)), np.ones((4, 3)))

    def test_equality(self):
        delta = np.zeros((2, 5))
        df = DisplacementField(delta, delta)

        with pytest.raises(TypeError, match="not DisplacementField"):
            df == "not a df"  # noqa

        assert df == df

    def test_norm(self):
        shape = (4, 5)

        delta_x_1 = np.zeros(shape)
        delta_y_1 = np.zeros(shape)

        delta_x_2 = np.ones(shape) * 3
        delta_y_2 = np.ones(shape) * 4

        df_1 = DisplacementField(delta_x_1, delta_y_1)
        df_2 = DisplacementField(delta_x_2, delta_y_2)

        assert np.allclose(df_1.norm, np.zeros(shape))
        assert np.allclose(df_2.norm, np.ones(shape) * 5)

    def test_transformation(self):
        """Check that going from displacement field to the actual
        transformation works perfectly.

        Notes
        -----
        Since we encore using float32 we need to compare with 1e-6 tolerance
        (float32 represents to 7 decimal digits)
        """

        shape = (10, 14)
        random_state = 111

        np.random.seed(random_state)

        delta_x = np.random.uniform(-3, 3, size=shape)
        delta_y = np.random.uniform(-3.4, 3.5, size=shape)

        df = DisplacementField(delta_x, delta_y)

        f_x, f_y = df.transformation

        x_1 = 2
        y_1 = 3
        x_2 = 6
        y_2 = 1

        assert f_x.dtype == np.float32
        assert f_y.dtype == np.float32

        assert f_x[y_1, x_1] == pytest.approx(x_1 + delta_x[y_1, x_1])
        assert f_y[y_1, x_1] == pytest.approx(y_1 + delta_y[y_1, x_1])
        assert f_x[y_2, x_2] == pytest.approx(x_2 + delta_x[y_2, x_2])
        assert f_y[y_2, x_2] == pytest.approx(y_2 + delta_y[y_2, x_2])


class TestFromFile:
    """Collections of tests focused on the classmethod `from_file`."""

    def test_wrong_input(self):
        """Input is wrong."""

        with pytest.raises(TypeError):
            DisplacementField.from_file(123213)

    def test_unsupported_suffix(self, tmpdir):
        """File with unsupported extension."""

        # create a file with unsupported extensions
        ext = "FAKE"
        full_path = pathlib.Path(str(tmpdir)) / "test_file.{}".format(ext)
        full_path.touch()

        with pytest.raises(ValueError):
            DisplacementField.from_file(full_path)

    @pytest.mark.parametrize(
        "as_str", [True, False], ids=["path_is_str", "path_is_Path"]
    )
    @pytest.mark.parametrize(
        "array",
        [
            np.ones((1, 2, 2), dtype="float32"),
            np.ones((1, 2), dtype="float32"),
            np.ones((1, 2, 3), dtype="float32"),
        ],
        ids=["correct", "wrong_dim", "wrong_channels"],
    )
    def test_npy(self, tmpdir, array, as_str):
        """Possible to read npy arrays."""
        full_path: Union[pathlib.Path, str] = (
            pathlib.Path(str(tmpdir)) / "file_temp.npy"
        )

        if as_str:
            full_path = str(full_path)

        np.save(full_path, array)

        if array.ndim == 3 and array.shape[2] == 2:
            df = DisplacementField.from_file(full_path)
            assert isinstance(df, DisplacementField)
            assert np.allclose(array[:, :, 0], df.delta_x)
            assert np.allclose(array[:, :, 1], df.delta_y)

        else:
            with pytest.raises(ValueError):
                DisplacementField.from_file(full_path)


class TestFromTransform:
    """A collection of tests focused on the class method ``from_transform``."""

    def test_incorrect_input_1(self):
        """Different shapes."""

        f_x = np.zeros((3, 4))
        f_y = np.zeros((2, 4))

        with pytest.raises(ValueError):
            DisplacementField.from_transform(f_x, f_y)

    def test_incorrect_input_2(self):
        """Additional dimensions."""

        f_x = np.zeros((2, 2, 4))
        f_y = np.zeros((2, 2, 4))

        with pytest.raises(ValueError):
            DisplacementField.from_transform(f_x, f_y)

    def test_equivalent(self):
        """Make sure that transform and displacement constructors are equivalent."""

        random_state = 10
        shape = (3, 4)

        np.random.seed(random_state)

        delta_x = np.random.randint(10, size=shape)
        delta_y = np.random.randint(10, size=shape)

        df = DisplacementField(delta_x, delta_y)

        df_new = DisplacementField.from_transform(*df.transformation)

        atol = 1e-4

        assert np.allclose(df.delta_x, df_new.delta_x, atol=atol)
        assert np.allclose(df.delta_y, df_new.delta_y, atol=atol)


class TestMultiplication:
    """A collection of tests focused on the __mul__ and __rmul__ method"""

    def test_correct_type(self):
        """Test that only works for numbers"""
        shape = (20, 23)
        c_1 = 123
        c_2 = 31241.1121
        c_3 = "3213"

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        assert isinstance(df * c_1, DisplacementField)
        assert isinstance(df * c_2, DisplacementField)

        with pytest.raises(TypeError):
            df * c_3

    def test_identity(self):
        """Test that does not affect identity mapping."""

        shape = (20, 23)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        assert df * 3 == df
        assert 1001.21 * df == df

    @pytest.mark.parametrize("c", [0, 1, 10, 123.3])
    def test_commutative(self, c):
        """Test that commutative"""

        shape = (20, 32)

        delta_x = np.random.uniform(0, 1, size=shape)
        delta_y = np.random.uniform(0, 1, size=shape)

        df = DisplacementField(delta_x, delta_y)

        assert df * c == c * df

    def test_correct_result(self):
        """Test that giving correct results."""

        c = 11.3

        delta_x = np.array([[1.2, 123], [13, 12.1]])

        delta_y = np.array([[121.2, 12], [13.88, 12.1]])

        df_mul = DisplacementField(delta_x, delta_y) * c

        assert np.allclose(delta_x * c, df_mul.delta_x)
        assert np.allclose(delta_y * c, df_mul.delta_y)


class TestSave:
    """Collection of tests focused on the `save` method."""

    def test_invalid_suffix(self, tmpdir):
        """Different suffix than .npy not accepted."""

        delta_x = np.random.random((2, 3))
        delta_y = np.random.random((2, 3))

        df_orig = DisplacementField(delta_x, delta_y)

        with pytest.raises(ValueError):
            df_orig.save(pathlib.Path(str(tmpdir)) / "fake.wrong")

    @pytest.mark.parametrize(
        "as_str", [True, False], ids=["filepath_is_str", "filepath_is_Path"]
    )
    def test_saveload(self, tmpdir, as_str):
        """Test that saving and loading loads the same dvf."""

        delta_x = np.random.random((2, 3))
        delta_y = np.random.random((2, 3))

        df_orig = DisplacementField(delta_x, delta_y)

        file_path: Union[pathlib.Path, str] = pathlib.Path(str(tmpdir)) / "test.npy"

        if as_str:
            file_path = str(file_path)

        df_orig.save(file_path)

        df_new = DisplacementField.from_file(file_path)

        assert df_orig == df_new


class TestWarp:
    """A collection of tests that are focused on the
    warp method of the DisplacementField."""

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_identity(self, img, interpolation, border_mode):
        """Make sure that zero DF (and therefore identity mapping)
        has no effect on the image."""

        shape = img.shape[:2]

        img_ = img.copy()
        img_[5, 5] = 1  # only one pixel has a nonzero intensity

        delta_x_dummy = np.zeros(shape)
        delta_y_dummy = np.zeros(shape)

        df_dummy = DisplacementField(delta_x_dummy, delta_y_dummy)
        img_warped = df_dummy.warp(
            img_, interpolation=interpolation, border_mode=border_mode
        )

        assert np.allclose(img_warped, img_)

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    @pytest.mark.parametrize("direction", ["U", "D", "L", "R"])
    def test_singe_pixel_shift(self, img_dummy, interpolation, border_mode, direction):
        """Test whether displacing a single pixel
        to a neighbouring pixel works as expected."""

        shape = img_dummy.shape
        img_dummy[5, 7] = 1  # only one pixel has a nonzero intensity

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        if direction == "U":
            delta_y[15, 7] = -10
            img_exp = np.zeros(shape)
            img_exp[[15, 5], 7] = 1

        elif direction == "D":
            delta_y[3, 7] = 2
            img_exp = np.zeros(shape)
            img_exp[[3, 5], 7] = 1

        elif direction == "L":
            delta_x[
                5, 10
            ] = (
                -3
            )  # When looking for  [5, 4], [5, 5] in the original picture will be used
            img_exp = np.zeros(shape)
            img_exp[5, [7, 10]] = 1

        elif direction == "R":
            delta_x[5, 6] = 1  #
            img_exp = np.zeros(shape)
            img_exp[5, [6, 7]] = 1

        else:
            raise ValueError("Unknown direction {}".format(direction))

        df_dummy = DisplacementField(delta_x, delta_y)
        img_warped = df_dummy.warp(
            img_dummy, interpolation=interpolation, border_mode=border_mode
        )  # dummy=moving, img_exp=reference

        assert np.allclose(img_warped, img_exp)

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_constant_transform(self, img, interpolation, border_mode):
        """Make sure that its possible to map all pixels
        from fixed into a single pixel in input."""

        shape = img.shape

        img_ = img.copy()

        r_star, c_star = 10, 13
        value_star = img_[r_star, c_star]

        x, y = np.meshgrid(list(range(shape[1])), list(range(shape[0])))

        delta_x = c_star - x
        delta_y = r_star - y

        df = DisplacementField(delta_x, delta_y)
        img_warped = df.warp(img_, interpolation=interpolation, border_mode=border_mode)

        assert np.allclose(np.ones(shape) * value_star, img_warped)

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_returns_new_array(self, img, interpolation, border_mode):
        """Test that new image is saved in a new array."""

        shape = img.shape[:2]

        delta = np.zeros(shape, dtype=np.float32)
        df = DisplacementField(delta, delta)

        img_warped = df.warp(img, interpolation=interpolation, border_mode=border_mode)

        assert not np.may_share_memory(img, img_warped)
        assert not np.shares_memory(img, img_warped)

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_dtype_conversions(self, img, interpolation, border_mode):
        """Test that dtype is conserved (at least for uint8 and float32)"""

        delta = np.zeros(img.shape, dtype=np.float32)
        df = DisplacementField(delta, delta)

        warped_img = df.warp(img, interpolation=interpolation, border_mode=border_mode)

        assert warped_img.dtype == img.dtype

    def test_invalid_names(self):
        shape = (10, 11)
        img = np.zeros(shape)

        delta = np.zeros(shape, dtype=np.float32)
        df = DisplacementField(delta, delta)

        with pytest.raises(KeyError):
            df.warp(img, interpolation="fake_interpolation")

        with pytest.raises(KeyError):
            df.warp(img, border_mode="fake_border_mode")


class TestWarpAnnotation:
    """Tests focused on `warp` method using nearest neigbours interpolation."""

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    def test_identity(self, dtype):
        """Identity has no effect"""

        shape = (4, 5)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        img = np.random.randint(300, size=shape).astype(
            dtype
        )  # might overflow to something but whatever
        res = df.warp(img, interpolation="nearest")

        assert np.all(res == img)
        assert img.dtype == res.dtype

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("random_state", range(30))
    def test_no_new_classes(self, random_state, dtype):
        """No creation of new classes - exactly one nearest neighbour."""

        shape = (9, 10)

        deltas = affine_simple(shape, rotation=np.pi / 10)
        df = DisplacementField(*deltas)
        np.random.seed(random_state)
        img = np.random.randint(300, size=shape).astype(
            dtype
        )  # might overflow to something but whatever

        img_warped = df.warp(img, interpolation="nearest", border_mode="replicate")

        assert set(np.unique(img_warped)).issubset(set(np.unique(img)))

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize(
        "translation_xy", [(-0.49, 0), (0.49, 0), (0, -0.49), (0, 0.49), (0.2, 0.2)]
    )
    def test_small_shift(self, translation_xy, dtype):
        """Test that small shifts do not affect the warp.

        Notes
        -----
        Anything that is withing a diameter of 1 (=radius of 0.5)
        should be assigned to the same point.
        """
        translation_x, translation_y = translation_xy
        shape = (9, 10)

        random_state = 1
        deltas = affine_simple(
            shape,
            translation_x=translation_x,
            translation_y=translation_y,
        )
        df = DisplacementField(*deltas)

        np.random.seed(random_state)
        img = np.random.randint(256, size=shape).astype(dtype)

        img_warped = df.warp(img, interpolation="nearest")

        assert np.all(img_warped == img)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    def test_outside_of_image(self, dtype):
        """The behavior should be replication of the border."""
        shape = (4, 5)

        delta_x = (
            np.ones(shape) * 100
        )  # basically all the images should be equal to the right border
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        img = np.random.randint(1, 256, size=shape).astype(dtype)

        output = df.warp(img, interpolation="nearest", border_mode="replicate")

        assert np.all(output == np.tile(img[:, [-1]], (1, shape[-1])))

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    def test_different_dtypes(self, dtype):
        """Create a rotation transformation and make sure dtypes do not matter."""
        power = int(dtype.partition("int")[-1])
        assert dtype[:3] == "int"

        delta_x = np.array([[1, 1, 0], [0, -1, -1]])

        delta_y = np.array([[0, 0, 1], [-1, 0, 0]])

        df = DisplacementField(delta_x, delta_y)

        lower_bound = 2 ** (power - 1 - 1) - 1
        upper_bound = 2 ** (power - 1) - 1

        img = np.random.randint(
            lower_bound, upper_bound, size=(2, 3), dtype=dtype
        ).astype(dtype)

        img[0, 0] = -img[0, 0]
        img[1, 1] = -img[1, 1]

        correct_output = np.array(
            [[img[0, 1], img[0, 2], img[1, 2]], [img[0, 0], img[1, 0], img[1, 1]]],
            dtype=dtype,
        )

        output = df.warp(img, interpolation="nearest")

        assert np.all(output == correct_output)
        assert output.dtype == correct_output.dtype


def test_affine():
    with pytest.raises(ValueError, match="correct is"):
        affine((3, 5), np.ones((3, 4)))

    matrix = np.eye(3)

    delta_x, delta_y = affine((5, 5), matrix)

    assert (delta_x == 0).all()
    assert (delta_y == 0).all()


@pytest.mark.parametrize("apply_centering", [True, False])
def test_affine_simple(apply_centering):
    delta_x, delta_y = affine_simple((5, 7), apply_centering=apply_centering)

    assert delta_x.shape == (5, 7)
    assert delta_y.shape == (5, 7)

    # Defaults lead to identity
    assert (delta_x == 0).all()
    assert (delta_y == 0).all()
