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
"""Different plotting routines."""
from typing import Iterable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from atldld.constants import REF_DIM_1UM
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
        to that returned by the `atldld.utils.get_corners_in_ref_space`
        function.

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
    # A semi-arbitrary choice of the reference space scale. This choice only
    # changes the ticks on the axes, but not the overall plot. The 25µm scale
    # is one of the common scales used for volumes.
    ref_space_scale = 25
    ref_space_size = np.array(REF_DIM_1UM) / ref_space_scale
    p, i, r = 0, 1, 2
    labels = {
        p: "p (coronal)",
        i: "i (transverse)",
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
    else:  # pragma: no cover
        raise NotImplementedError(f"Unknown plane of section: {plane_of_section}")

    # The figure width is fixed and arbitrary. (Is there a more clever choice?)
    # The figure height is computed to match the ratio between the total width
    # and height of all subplots. This way the ratios of the x and y axes are
    # roughly the same (but not quite since the in-between spaces, titles, etc.
    # are not taken into account...)
    plot_width = sum(ref_space_size[x_axis] for x_axis in x_axes)
    plot_height = ref_space_size[y_axis]
    fig_width_inches = 14
    fig_height_inches = fig_width_inches * plot_height / plot_width

    fig = Figure(figsize=(fig_width_inches, fig_height_inches))
    fig.set_tight_layout(True)
    # The width ratios of subplots are based on the reference volume dimensions,
    # this way the scales of the x-axes roughly match.
    axs = fig.subplots(
        ncols=4,
        sharey=True,
        gridspec_kw={"width_ratios": [ref_space_size[x_axis] for x_axis in x_axes]},
    )
    # Y-label only on the left-most plot because it's the same for all plots
    axs[0].set_ylabel(labels[y_axis], fontsize=16)

    # Add the legend for the reference space lines
    ref_space_line_style = {"color": "blue", "linestyle": ":"}
    line = Line2D([], [], **ref_space_line_style)
    axs[0].legend(
        [line],
        ["Reference space boundary"],
        loc="upper left",
        bbox_to_anchor=(0, -0.2),
        borderaxespad=0,
        frameon=False,
    )

    # The actual plotting
    for ax, edge, x_axis, invert, title in zip(axs, edges, x_axes, inverts, titles):
        # Axes setup
        ax.grid(True, linestyle=":", color="gray")
        ax.set_title(title)
        ax.set_xlabel(labels[x_axis], fontsize=16)

        # Reference space boundary lines
        ax.axvline(0, **ref_space_line_style)
        ax.axvline(ref_space_size[x_axis], **ref_space_line_style)
        ax.axhline(0, **ref_space_line_style)
        ax.axhline(ref_space_size[y_axis], **ref_space_line_style)

        # Plot the section image edges
        for corners in all_corners:
            points = corners[np.ix_(edge, [x_axis, y_axis])]
            coords = points.T / ref_space_scale
            ax.plot(*coords, color="green")
            ax.scatter(*coords, color="red")
        if invert:
            ax.invert_xaxis()

    return fig
