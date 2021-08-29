"""Mathematical computations."""
from typing import Sequence

import numpy as np


def find_shearless_3d_affine(
        p_from: Sequence[np.ndarray],
        p_to: Sequence[np.ndarray]
) -> np.ndarray:
    """Find a 3D shearless affine transformation given the mapping of 3 points.

    Parameters
    ----------
    p_from
        Three input points in a 3-dimensional Euclidean space.
    p_to
        Three output point in a 3-dimensional Euclidean space.

    Returns
    -------
    np.ndarray
        The affine transform that maps ``p_from`` to ``p_to``.

    With the assumption of no shear we can find the 4th point and its mapping
    by the cross product. For example ``p_from = (p1, p2, p3)`` gives two
    vectors ``v1 = p2 - p1`` and ``v2 = p3 - p1``, giving ``v3 = v1 x v2``. The
    fourth point is then given by ``p4 = p1 + v3``.

    The only caveat is the length of the vector ``v3``. We must make sure that
    it scales in the same way between ``p_from`` and ``p_to`` as all other
    points. The easiest way to ensure this is to give it the same length as the
    ``v1`` vector: ``v3 = |v1| * (v1 x v2 / |v1| / |v2|) = v1 x v2 / |v2|``.
    """
    # TODO
    # Some sanity checks on the input data: make sure the two triangles
    # formed by the input points have the same shape and orientation

    def add_fourth(points):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = np.cross(v1, v2) / np.linalg.norm(v2)
        p4 = points[0] + v3

        return np.stack([*points, p4])

    p_from = add_fourth(p_from)
    p_to = add_fourth(p_to)

    # check uniform scaling
    # from itertools import combinations
    # scales = []
    # tolerance = 0.02
    # for i, j in combinations(range(3), 2):
    #     len_from = np.linalg.norm(p_from[i] - p_from[j])
    #     len_to = np.linalg.norm(p_to[i] - p_to[j])
    #     scales.append(len_from / len_to)
    #     print(scales)
    #     print(np.all([
    #         abs(s1 - s2) / min(s1, s2) < tolerance
    #         for s1, s2 in combinations(scales, 2)
    #     ]))

    # Add homogenous coordinate and transpose so that
    # - dim_0 = "xyz1"
    # - dim_1 = points
    p_from = np.concatenate([p_from.T, np.array([[1, 1, 1, 1]])])
    p_to = np.concatenate([p_to.T, np.array([[1, 1, 1, 1]])])

    # Compute the affine transform so that p_to = A @ p_from
    # => A = p_to @ p_from_inv
    return p_to @ np.linalg.inv(p_from)
