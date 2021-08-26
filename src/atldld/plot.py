"""Different plotting routines."""
from typing import Iterable

import numpy as np
from matplotlib.figure import Figure

from atldld import constants
from atldld.dataset import PlaneOfSection


def preview_sagittal_dataset(
    all_corners: Iterable[np.ndarray],
    plane_of_section: PlaneOfSection,
) -> Figure:
    """Plot a preview of how section images fit into the reference space.

    Parameters
    ----------
    all_corners
        The corners of all section images. Each element in this iterable should
        be a NumPy array of shape (4, 3). The format of this array corresponds
        to that returned by the `atldld.requests.get_ref_corners` function.

        The first axis refers to the four corners of a section image in the
        following order:

        1. Lower left (0, 0)
        2. Lower right (0, 1)
        3. Upper right (1, 1)
        4. Upper left (1, 0)

        This corresponds to following the corners counterclockwise starting with
        the corner in the axes origin. The second array axis contains the 3D
        coordinates of the corners in the standard PIR references space.

    plane_of_section
        The plane of section of the dataset. Can be either coronal or sagittal.

    Returns
    -------
    fig
        The figure with the plot.
    """
    scale = 25
    n_p, n_i, n_r = np.array(constants.REF_DIM_1UM) / scale
    p, i, r = 0, 1, 2

    fig = Figure(figsize=(14, 4))
    fig.set_tight_layout(True)
    axs = fig.subplots(
        ncols=4,
        sharey=True,
        gridspec_kw={"width_ratios": [16 / 7, 1, 16 / 7, 1]},
    )
    for ax in axs.ravel():
        ax.grid(True, linestyle=":", color="gray")
        ax.set_ylim((0, n_r))
    ax1, ax2, ax3, ax4 = axs.ravel()

    def draw_slice_2d(ax, points):
        coords = points.T
        ax.plot(*coords, color="green")
        ax.scatter(*coords, color="red")

    ax1.set_title("$-i$")
    ax1.set_xlabel("p (coronal)", fontsize=16)
    ax1.set_ylabel("r (sagittal)", fontsize=16)
    ax1.axvline(0, color="blue", linestyle=":")
    ax1.axvline(n_p, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax1, corners[np.ix_([0, 1], [p, r])] / scale)

    ax2.set_title("$-p$")
    ax2.set_xlabel("i (transversal)", fontsize=16)
    ax2.axvline(0, color="blue", linestyle=":")
    ax2.axvline(n_i, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax2, corners[np.ix_([1, 2], [i, r])] / scale)

    ax3.set_title("$+i$")
    ax3.set_xlabel("p (coronal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax3, corners[np.ix_([2, 3], [p, r])] / scale)
    ax3.axvline(0, color="blue", linestyle=":")
    ax3.axvline(n_p, color="blue", linestyle=":")
    ax3.invert_xaxis()

    ax4.set_title("$+p$")
    ax4.set_xlabel("i (transversal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax4, corners[np.ix_([3, 0], [i, r])] / scale)
    ax4.axvline(0, color="blue", linestyle=":")
    ax4.axvline(n_i, color="blue", linestyle=":")
    ax4.invert_xaxis()

    return fig
