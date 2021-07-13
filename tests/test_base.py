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
from unittest.mock import Mock

import numpy as np
import pytest
from skimage.transform import resize

from atldld.base import DisplacementField

SUPPORTED_DTYPES_ANNOTATIONS = ["int8", "int16", "int32"]
SUPPORTED_APPROACHES_ANNOTATIONS = ["scipy", "opencv"]
WARP_INTERPOLATIONS = ["linear"]
WARP_BORDER_MODES = ["constant", "replicate"]


class TestAdjust:
    """A collection of tests focused on the adjust method.
    Notes
    -----
    We cannot really use __equal__ here because of some weird precision issues.
    """

    @staticmethod
    def eps_equal(lhs, rhs, eps=1e-6):
        return np.all(abs(lhs.delta_x - rhs.delta_x) < eps) and np.all(
            abs(lhs.delta_y - rhs.delta_y) < eps
        )

    def test_inactive(self):
        """Create a wild transform without leaving the image space."""

        random_state = 55
        shape = (_, w) = (40, 30)

        n_pixels = np.prod(shape)
        np.random.seed(random_state)
        perm = np.random.permutation(n_pixels)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        for i, x in enumerate(perm):
            r_inp, c_inp = i // w, i % w
            r_out, c_out = x // w, x % w

            delta_x[r_inp, c_inp] = c_out - c_inp
            delta_y[r_inp, c_inp] = r_out - r_inp

        df = DisplacementField(delta_x, delta_y)
        assert df == df.adjust()

        delta_x_new = delta_x.copy()
        delta_x_new[0, 0] = -1

        df_new = DisplacementField(delta_x_new, delta_y)
        assert df_new != df_new.adjust()

    def test_correct_scale(self):
        """Create outsider pixels on purpose with clear shrunk vectors."""

        shape = (h, w) = (40, 30)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)
        delta_x_correct = np.zeros(shape)
        delta_y_correct = np.zeros(shape)

        # Right
        delta_x[0, 0] = 100
        delta_x_correct[0, 0] = w - 1 - 0

        # Up
        delta_y[15, 10] = -100
        delta_y_correct[15, 10] = -15

        # Down
        delta_y[25, 11] = 100000
        delta_y_correct[25, 11] = h - 1 - 25

        # Left
        delta_x[1, 13] = -131231
        delta_x_correct[1, 13] = -13

        # random_1
        delta_x[2, 4] = 10
        delta_y[2, 4] = 100
        c_1 = (h - 1 - 2) / 100
        delta_x_correct[2, 4] = c_1 * 10
        delta_y_correct[2, 4] = c_1 * 100

        df = DisplacementField(delta_x, delta_y)

        df_correct = DisplacementField(delta_x_correct, delta_y_correct)

        assert TestAdjust.eps_equal(df.adjust(force_inside_border=True), df_correct)


class TestCompose:
    """A collection of tests focused on the composition __call__ method."""

    @staticmethod
    def eps_equal(lhs, rhs, eps=1e-6):
        return np.all(abs(lhs.delta_x - rhs.delta_x) < eps) and np.all(
            abs(lhs.delta_y - rhs.delta_y) < eps
        )

    def test_id_compose_id_gives_id(self):
        """Just test that composing identities gives an identity."""
        shape = (300, 200)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        # I(I(x)) = I(x)
        assert df == df(df)  # __eq__ implemented

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_id_compose_whatever_gives_whatever(self, interpolation, border_mode):
        """Test that identity(whatever) = whatever and whatever(identity) = whatever"""

        shape = (20, 30)

        delta_x_identity = np.zeros(shape)
        delta_y_identity = np.zeros(shape)

        delta_x_whatever = np.random.randint(-5, 5, size=shape)
        delta_y_whatever = np.random.randint(-5, 5, size=shape)

        df_identity = DisplacementField(delta_x_identity, delta_y_identity)
        df_whatever = DisplacementField(
            delta_x_whatever, delta_y_whatever
        ).adjust()  # make sure doe

        df_comp_1 = df_identity(
            df_whatever, interpolation=interpolation, border_mode=border_mode
        )
        df_comp_2 = df_whatever(
            df_identity, interpolation=interpolation, border_mode=border_mode
        )

        eps = 0.1
        assert TestCompose.eps_equal(df_comp_1, df_whatever, eps=eps)
        assert TestCompose.eps_equal(df_comp_2, df_whatever, eps=eps)
        # assert TestCompose.eps_equal(df_comp_2, df_whatever, eps=1e-6)  # works

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    @pytest.mark.parametrize("random_state", [0, 1, 10])
    def test_no_interpolation(self, random_state, interpolation, border_mode):
        """Designs an inner mapping in a way that
        it always ends on a grid so there is no need for interpolation

        Notes
        -----
        delta_comp(x) = ... = delta_outer(x + delta_inner(x)) + delta_inner(x)
        """

        eps = 1e-6
        shape = (5, 6)

        np.random.seed(random_state)

        # The inner mapping always ends up EXACTLY on grid points
        # no interpolation needed
        delta_x_inner = np.zeros(shape)
        delta_y_inner = np.zeros(shape)

        for r in range(shape[0]):
            for c in range(shape[1]):
                delta_x_inner[r, c] = np.random.randint(-c, shape[1] - c)
                delta_y_inner[r, c] = np.random.randint(-r, shape[0] - r)

        delta_x_outer = np.random.random(size=shape)
        delta_y_outer = np.random.random(size=shape)

        df_inner = DisplacementField(delta_x_inner, delta_y_inner)
        df_outer = DisplacementField(delta_x_outer, delta_y_outer)

        df_comp = df_outer(
            df_inner, interpolation=interpolation, border_mode=border_mode
        )

        for r in range(shape[0]):
            for c in range(shape[1]):
                new_coords = int(r + delta_y_inner[r, c]), int(c + delta_x_inner[r, c])

                assert (
                    abs(
                        delta_x_outer[new_coords]
                        + delta_x_inner[r, c]
                        - df_comp.delta_x[r, c]
                    )
                    < eps
                )
                assert (
                    abs(
                        delta_y_outer[new_coords]
                        + delta_y_inner[r, c]
                        - df_comp.delta_y[r, c]
                    )
                    < eps
                )

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_simple(self, interpolation, border_mode):
        """Chain two almost identity transformations."""

        shape = (40, 50)

        delta_x_inner = np.zeros(shape)
        delta_y_inner = np.zeros(shape)
        delta_x_inner[4, 9] = 2
        delta_y_inner[4, 9] = 5

        df_inner = DisplacementField(delta_x_inner, delta_y_inner)

        delta_x_outer = np.zeros(shape)
        delta_y_outer = np.zeros(shape)
        delta_x_outer[9, 11] = 10
        delta_y_outer[9, 11] = 20

        df_outer = DisplacementField(delta_x_outer, delta_y_outer)

        delta_x_res_exp = np.zeros(shape)
        delta_x_res_exp[4, 9] = 2 + 10
        delta_x_res_exp[9, 11] = 10

        delta_y_res_exp = np.zeros(shape)
        delta_y_res_exp[4, 9] = 5 + 20
        delta_y_res_exp[9, 11] = 20

        df_exp = DisplacementField(delta_x_res_exp, delta_y_res_exp)
        df_comp = df_outer(
            df_inner, interpolation=interpolation, border_mode=border_mode
        )

        assert df_exp == df_comp

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_composition_intermediate(self, interpolation, border_mode):
        """Test that composing two opposite
        translations results in an identity (almost)"""

        shape = (400, 200)

        matrix_1 = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        matrix_2 = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])
        matrix_id = np.eye(3)

        df_1 = DisplacementField.generate(shape, approach="affine", matrix=matrix_1)
        df_2 = DisplacementField.generate(shape, approach="affine", matrix=matrix_2)

        df_id = DisplacementField.generate(shape, approach="affine", matrix=matrix_id)

        res = df_2(df_1, interpolation=interpolation, border_mode=border_mode)

        # Works only inside:D
        assert np.allclose(res.delta_x[2:-2, 2:-2], df_id.delta_x[2:-2, 2:-2])
        assert np.allclose(res.delta_y[2:-2, 2:-2], df_id.delta_y[2:-2, 2:-2])


class TestDisplacementFieldEssentials:
    """A collection of tests that focus on the
    fundamental functioning of the DisplacementField class.

    Notes
    -----
    Mostly focused on the properties.
    """

    def test_average_displacement(self):
        """Test focused on the `average_displacement` property"""

        shape = (4, 4)

        delta_x_1 = np.zeros(shape)
        delta_y_1 = np.zeros(shape)

        delta_x_1[0, 0] = 2

        delta_x_2 = np.zeros(shape)
        delta_y_2 = np.zeros(shape)

        df_1 = DisplacementField(delta_x_1, delta_y_1)
        df_2 = DisplacementField(delta_x_2, delta_y_2)

        assert df_1.average_displacement == 2 / 16
        assert df_2.average_displacement == 0

    def test_is_valid(self):
        """Focused on the is_valid property"""
        shape = (3, 4)

        # 1 - valid
        delta_x_1 = np.zeros(shape)
        delta_y_1 = np.zeros(shape)
        df_1 = DisplacementField(delta_x_1, delta_y_1)

        # 2 - invalid
        delta_x_2 = np.zeros(shape)
        delta_y_2 = np.zeros(shape)
        delta_x_2[0, 1] = np.nan
        df_2 = DisplacementField(delta_x_2, delta_y_2)

        # 3 - valid
        delta_x_3 = np.zeros(shape)
        delta_y_3 = np.zeros(shape)
        delta_x_3[0, 2] = np.inf
        df_3 = DisplacementField(delta_x_3, delta_y_3)

        assert df_1.is_valid
        assert not df_2.is_valid
        assert not df_3.is_valid

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

    def test_outsiders(self):
        """Test that number of outsiders is computed correctly."""

        shape = (h, w) = (20, 40)

        delta_y_all = np.ones(shape) * (h + 1)
        delta_x_all = np.ones(shape) * (w + 1)

        delta_y_none = np.zeros(shape)
        delta_x_none = np.zeros(shape)

        delta_y_one = np.zeros(shape)
        delta_x_one = np.zeros(shape)
        delta_x_one[10, 10] = w + 1

        df_all = DisplacementField(delta_x_all, delta_y_all)
        df_none = DisplacementField(delta_x_none, delta_y_none)
        df_one = DisplacementField(delta_x_one, delta_y_one)

        mask_all = np.ones(shape, dtype=bool)
        mask_none = np.zeros(shape, dtype=bool)
        mask_one = np.zeros(shape, dtype=bool)
        mask_one[10, 10] = True

        assert np.all(df_all.outsiders == mask_all)
        assert np.all(df_none.outsiders == mask_none)
        assert np.all(df_one.outsiders == mask_one)

    def test_scaled_delta_xy(self):
        """The goal is to make sure that one can access
        scaled versions of delta_x and delta_y"""

        shape = (h, w) = (200, 400)

        delta_x = np.ones(shape, dtype=np.float32) * 5
        delta_y = np.ones(shape, dtype=np.float32) * 8

        df = DisplacementField(delta_x, delta_y)

        assert np.allclose(df.delta_x_scaled, (delta_x / w)) and np.allclose(
            df.delta_y_scaled, (delta_y / h)
        )

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

    def test_mask(self):
        shape = (300, 500)

        delta_x = np.ones(shape) * 11
        delta_y = np.ones(shape) * 8
        df = DisplacementField(delta_x, delta_y)

        mask_matrix_wrong_shape = np.zeros((shape[0], shape[1] + 1), dtype=bool)
        mask_matrix_wrong_type = np.zeros(shape)
        mask_matrix = np.zeros(shape, dtype=bool)
        mask_matrix[9:11, 18:24] = True

        with pytest.raises(ValueError):
            df.mask(mask_matrix_wrong_shape)

        with pytest.raises(TypeError):
            df.mask(mask_matrix_wrong_type)

        df_masked = df.mask(mask_matrix, fill_value=(155, 12))

        assert np.all(
            df_masked.delta_x[mask_matrix] == np.ones(shape)[mask_matrix] * 11
        )
        assert np.all(df_masked.delta_y[mask_matrix] == np.ones(shape)[mask_matrix] * 8)
        assert np.all(
            df_masked.delta_x[~mask_matrix] == np.ones(shape)[~mask_matrix] * 155
        )
        assert np.all(
            df_masked.delta_y[~mask_matrix] == np.ones(shape)[~mask_matrix] * 12
        )


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
        full_path = pathlib.Path(str(tmpdir)) / "file_temp.npy"

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


class TestJacobian:
    """A collection of tests focused on the `jacobian` property."""

    def test_identity(self):
        """Jacobian of identity has at each point determinant 1."""

        shape = (4, 5)
        df = DisplacementField(np.zeros(shape), np.zeros(shape))

        assert np.all(df.jacobian == np.ones(shape))

    def test_one_point_collapse(self):
        """Mapping all points into 1 point results in a noninvertible mapping."""

        shape = (3, 4)

        collapse_point = (2, 2)  # r, c

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        for r in range(shape[0]):
            for c in range(shape[1]):
                delta_x[r, c] = collapse_point[1] - c
                delta_y[r, c] = collapse_point[0] - r

        df = DisplacementField(delta_x, delta_y)

        assert np.allclose(df.jacobian, 0)


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


class TestPlotRanges:
    """Tests focused on the ``plot_ranges`` method."""

    def test_basic(self, monkeypatch):
        shape = (10, 11)
        df = DisplacementField.generate(shape, approach="identity")

        fake_ax = Mock()
        fake_plt = Mock()
        fake_plt.subplots = lambda *args, **kwargs: (None, fake_ax)

        monkeypatch.setattr("atldld.base.plt", fake_plt)

        df.plot_ranges()

        assert fake_ax.scatter.call_count > 0
        assert fake_ax.legend.call_count > 0


class TestResize:
    """A collection of tests focused on the resize method."""

    def test_incorrect_input(self):
        """Test that wrong inputs lead to an error."""

        shape = (40, 10)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        with pytest.raises(ValueError):
            df.resize((1, 1, 2))

        with pytest.raises(TypeError):
            df.resize("some_str")

    @pytest.mark.parametrize("new_shape", [(20, 25), (50, 40), (23, 33), (42, 20)])
    def test_identity(self, new_shape):
        """Resized dvf has the correct shape."""
        shape = (40, 30)  # Needs to stay like this

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        assert df.resize(new_shape).shape == new_shape

    @pytest.mark.parametrize("h_delta", [-50, 0, 100])
    @pytest.mark.parametrize("w_delta", [-50, 0, 100])
    def test_warp_invariance(self, img, h_delta, w_delta):
        """Test that df_resized.warp(img) ~ resized(df.warp(img))."""
        shape = img.shape
        new_shape = (shape[0] + h_delta, shape[1] + w_delta)

        assert (
            new_shape[0] > 0 and new_shape[1] > 0
        )  # if not satisfied fixture image too small

        # Zoom in
        df = DisplacementField.generate(shape, approach="affine_simple", scale_x=0.8)

        df_resized = df.resize(new_shape)
        img_1 = df_resized.warp(img)
        img_2 = resize(df.warp(img), new_shape, preserve_range=True)  # preserving dtype

        # WARNING MAKE SURE THAT DOES NOT RUN INTO OVERFLOW FOR UINT8
        adiff = abs(np.subtract(img_1, img_2, dtype="float"))

        if img.dtype == np.uint8:
            eps = 25
        elif img.dtype == np.float32:
            eps = 1e-1
        else:
            raise ValueError("This test is only made for uint8 and float32 images.")

        assert np.median(adiff) < eps


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

        file_path = pathlib.Path(str(tmpdir)) / "test.npy"

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

        df = DisplacementField.generate(shape, approach="identity")

        img_warped = df.warp(img, interpolation=interpolation, border_mode=border_mode)

        assert not np.may_share_memory(img, img_warped)
        assert not np.shares_memory(img, img_warped)

    @pytest.mark.parametrize("interpolation", WARP_INTERPOLATIONS)
    @pytest.mark.parametrize("border_mode", WARP_BORDER_MODES)
    def test_dtype_conversions(self, img, interpolation, border_mode):
        """Test that dtype is conserved (at least for uint8 and float32)"""

        df = DisplacementField.generate(img.shape, approach="identity")

        warped_img = df.warp(img, interpolation=interpolation, border_mode=border_mode)

        assert warped_img.dtype == img.dtype

    def test_invalid_names(self):
        shape = (10, 11)
        img = np.zeros(shape)

        df = DisplacementField.generate(shape, approach="identity")

        with pytest.raises(KeyError):
            df.warp(img, interpolation="fake_interpolation")

        with pytest.raises(KeyError):
            df.warp(img, border_mode="fake_border_mode")


class TestWarpAnnotation:
    """A collection of tests focused on the `warp_annotation` method."""

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("approach", SUPPORTED_APPROACHES_ANNOTATIONS)
    def test_identity(self, approach, dtype):
        """Identity has no effect"""

        shape = (4, 5)

        delta_x = np.zeros(shape)
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        img = np.random.randint(300, size=shape).astype(
            dtype
        )  # might overflow to something but whatever
        res = df.warp_annotation(img, approach=approach)

        assert np.all(res == img)
        assert img.dtype == res.dtype

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("approach", SUPPORTED_APPROACHES_ANNOTATIONS)
    @pytest.mark.parametrize("random_state", range(30))
    def test_no_new_classes(self, random_state, approach, dtype):
        """No creation of new classes - exactly one nearest neighbour."""

        shape = (9, 10)

        df = DisplacementField.generate(
            shape, approach="affine_simple", rotation=np.pi / 10
        )

        np.random.seed(random_state)
        img = np.random.randint(300, size=shape).astype(
            dtype
        )  # might overflow to something but whatever

        img_warped = df.warp_annotation(img, approach=approach)

        assert set(np.unique(img_warped)).issubset(set(np.unique(img)))

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("approach", SUPPORTED_APPROACHES_ANNOTATIONS)
    @pytest.mark.parametrize(
        "translation_xy", [(-0.49, 0), (0.49, 0), (0, -0.49), (0, 0.49), (0.2, 0.2)]
    )
    def test_small_shift(self, translation_xy, approach, dtype):
        """Test that small shifts do not affect the warp.

        Notes
        -----
        Anything that is withing a diameter of 1 (=radius of 0.5)
        should be assigned to the same point.
        """
        translation_x, translation_y = translation_xy
        shape = (9, 10)

        random_state = 1
        df = DisplacementField.generate(
            shape,
            approach="affine_simple",
            translation_x=translation_x,
            translation_y=translation_y,
        )

        np.random.seed(random_state)
        img = np.random.randint(256, size=shape).astype(dtype)

        img_warped = df.warp_annotation(img, approach=approach)

        assert np.all(img_warped == img)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("approach", SUPPORTED_APPROACHES_ANNOTATIONS)
    def test_outside_of_image(self, approach, dtype):
        """The behavior should be replication of the border."""
        shape = (4, 5)

        delta_x = (
            np.ones(shape) * 100
        )  # basically all the images should be equal to the right border
        delta_y = np.zeros(shape)

        df = DisplacementField(delta_x, delta_y)

        img = np.random.randint(1, 256, size=shape).astype(dtype)

        output = df.warp_annotation(img, approach=approach)

        assert np.all(output == np.tile(img[:, [-1]], (1, shape[-1])))

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    @pytest.mark.parametrize("approach", SUPPORTED_APPROACHES_ANNOTATIONS)
    def test_different_dtypes(self, dtype, approach):
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

        output = df.warp_annotation(img, approach=approach)

        assert np.all(output == correct_output)
        assert output.dtype == correct_output.dtype

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES_ANNOTATIONS)
    def test_approaches_equivalent(self, dtype):
        """Make sure approaches equivalent"""
        shape = (10, 11)

        img = np.random.randint(1, 256, size=shape).astype(dtype)
        df = DisplacementField.generate(
            shape, approach="affine_simple", rotation=np.pi / 10
        )

        all_results = [
            df.warp_annotation(img, approach=x)
            for x in SUPPORTED_APPROACHES_ANNOTATIONS
        ]

        for i in range(len(all_results) - 1):
            assert np.array_equal(all_results[i], all_results[i + 1])


@pytest.mark.parametrize(
    "approach",
    [
        "affine",
        "affine_simple",
        "identity",
    ],
)
def test_construction(approach):
    """Just check default factory methods are able to construct the class."""

    shape = (500, 500)
    inst = DisplacementField.generate(shape, approach=approach)

    assert isinstance(inst, DisplacementField)
