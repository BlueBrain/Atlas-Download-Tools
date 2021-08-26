"""Different plotting routines."""
from typing import Iterable

import numpy as np
from matplotlib.figure import Figure

from atldld.constants import REF_DIM_25UM
from atldld.dataset import PlaneOfSection


def dataset_preview(
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
    ref_space_size = np.array(REF_DIM_25UM)
    p, i, r = 0, 1, 2
    labels = {
        p: "p (coronal)",
        i: "i (transversal)",
        r: "r (sagittal)",
    }
    # We'll plot the views of all four edges in counterclockwise order starting
    # with the bottom edge. The last two x-axes are inverted so that the edge
    # vertices on the right of a plot appear on the left of the following plot.
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    titles = ["Bottom Edges", "Right Edges", "Top Edges", "Left Edges"]
    inverts = [False, False, True, True]

    # Depending on the plane of section the views are different
    if plane_of_section == PlaneOfSection.CORONAL:
        y_axis = p
        x_axes = [r, i, r, i]
    elif plane_of_section == PlaneOfSection.SAGITTAL:
        y_axis = r
        x_axes = [p, i, p, i]
    else:
        raise NotImplementedError(f"Unknown plane of section: {plane_of_section}")

    # Figure size is arbitrary, maybe make it more clever at some point? The
    # width ratios are based on the reference volume dimensions, this way the
    # scales of the x-axes roughly match.
    fig = Figure(figsize=(14, 4))
    fig.set_tight_layout(True)
    axs = fig.subplots(
        ncols=4,
        sharey=True,
        gridspec_kw={
            "width_ratios": [ref_space_size[x_axis] for x_axis in x_axes]
        },
    )
    # Y-label only on the left-most plot because it's the same for all plots
    axs[0].set_ylabel(labels[y_axis], fontsize=16)

    # The actual plotting
    for ax, edge, x_axis, invert, title in zip(axs, edges, x_axes, inverts, titles):
        ax.grid(True, linestyle=":", color="gray")
        ax.set_ylim((0, ref_space_size[y_axis]))
        ax.set_title(title)
        ax.set_xlabel(labels[x_axis], fontsize=16)
        ax.axvline(0, color="blue", linestyle=":")
        ax.axvline(ref_space_size[x_axis], color="blue", linestyle=":")
        for corners in all_corners:
            points = corners[np.ix_(edge, [x_axis, y_axis])]
            coords = points.T / 25
            ax.plot(*coords, color="green")
            ax.scatter(*coords, color="red")
        if invert:
            ax.invert_xaxis()

    return fig
