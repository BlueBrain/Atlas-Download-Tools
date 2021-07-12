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

import numpy as np
import pytest


@pytest.fixture(scope="function")
def img_dummy():
    """Generate a dummy image made out of all zeros."""
    return np.zeros((20, 30), dtype=np.float32)


@pytest.fixture(scope="function")
def img():
    """Generate a grayscale image with dtype float32."""
    img = np.random.rand(200, 300)
    return img.astype(np.float32)
