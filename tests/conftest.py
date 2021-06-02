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
