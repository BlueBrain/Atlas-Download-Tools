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
"""Define all fixtures."""

import json
import pathlib

import numpy as np
import pytest

DATA_FOLDER = pathlib.Path(__file__).parent / "data"
PIR_TO_XY_FOLDER = DATA_FOLDER / "sync" / "pir_to_xy"
PIR_TO_XY_RESPONSES = sorted(PIR_TO_XY_FOLDER.iterdir())
XY_TO_PIR_FOLDER = DATA_FOLDER / "sync" / "xy_to_pir"
XY_TO_PIR_RESPONSES = sorted(XY_TO_PIR_FOLDER.iterdir())


@pytest.fixture(autouse=True)
def custom_cache_dir(monkeypatch, tmpdir):
    # Automatically use a custom cache directory for all tests to avoid writing
    # cache data to the user's real cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmpdir))


@pytest.fixture(scope="function")
def img_dummy():
    """Generate a dummy image made out of all zeros."""
    return np.zeros((20, 30), dtype=np.float32)


@pytest.fixture(scope="function")
def img():
    """Generate a grayscale image with dtype float32."""
    img = np.random.rand(200, 300)
    return img.astype(np.float32)


@pytest.fixture(
    scope="session",
    params=PIR_TO_XY_RESPONSES,
    ids=[p.stem for p in PIR_TO_XY_RESPONSES],
)
def pir_to_xy_response(request):
    path = request.param

    with path.open() as f:
        data = json.load(f)

    return data


@pytest.fixture(
    scope="session",
    params=XY_TO_PIR_RESPONSES,
    ids=[p.stem for p in XY_TO_PIR_RESPONSES],
)
def xy_to_pir_response(request):
    path = request.param

    with path.open() as f:
        data = json.load(f)

    return data
