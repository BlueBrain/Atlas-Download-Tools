import numpy as np
from matplotlib.figure import Figure

from atldld import constants


def draw_slice_2d(ax, points):
    coords = points.T
    ax.plot(*coords, color="green")
    ax.scatter(*coords, color="red")


def preview_sagittal_dataset(all_corners):
    scale = 25
    n_p, n_i, n_r = np.array(constants.REF_DIM_1UM) / scale
    p, i, r = 0, 1, 2

    fig = Figure(figsize=(14, 4), dpi=200)
    fig.set_tight_layout(True)
    axs = fig.subplots(
        ncols=4,
        sharey=True,
        gridspec_kw={"width_ratios": [16 / 7, 1, 16 / 7, 1]}
    )
    for ax in axs.ravel():
        ax.grid(True, linestyle=":", color="gray")
        ax.set_ylim((0, n_r))
    #     ax.set_aspect("equal")
    ax1, ax2, ax3, ax4 = axs.ravel()

    ax1.set_title("Edge $(0, 0)-(1, 0)$")
    ax1.set_xlabel("p (coronal)", fontsize=16)
    ax1.set_ylabel("r (sagittal)", fontsize=16)
    ax1.axvline(0, color="blue", linestyle=":")
    ax1.axvline(n_p, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax1, corners[np.ix_([0, 1], [p, r])] / scale)

    ax2.set_title("Edge $(1, 0)-(1, 1)$")
    ax2.set_xlabel("i (transversal)", fontsize=16)
    ax2.axvline(0, color="blue", linestyle=":")
    ax2.axvline(n_i, color="blue", linestyle=":")
    for corners in all_corners:
        draw_slice_2d(ax2, corners[np.ix_([1, 2], [i, r])] / scale)

    ax3.set_title("Edge $(1, 1)-(0, 1)$")
    ax3.set_xlabel("p (coronal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax3, corners[np.ix_([2, 3], [p, r])] / scale)
    ax3.axvline(0, color="blue", linestyle=":")
    ax3.axvline(n_p, color="blue", linestyle=":")
    ax3.invert_xaxis()

    ax4.set_title("Edge $(0, 1)-(0, 0)$")
    ax4.set_xlabel("i (transversal)", fontsize=16)
    for corners in all_corners:
        draw_slice_2d(ax4, corners[np.ix_([3, 0], [i, r])] / scale)
    ax4.axvline(0, color="blue", linestyle=":")
    ax4.axvline(n_i, color="blue", linestyle=":")
    ax4.invert_xaxis()

    return fig
