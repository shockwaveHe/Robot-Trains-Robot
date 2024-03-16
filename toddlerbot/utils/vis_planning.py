import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from toddlerbot.utils.vis_utils import *


def draw_foot(ax, center, size, angle, side):
    length, width = size
    # Calculate the corner points
    dx = np.cos(angle) * length / 2
    dy = np.sin(angle) * length / 2
    corners = np.array(
        [
            [
                center[0] - dx - width * np.sin(angle) / 2,
                center[1] - dy + width * np.cos(angle) / 2,
            ],
            [
                center[0] + dx - width * np.sin(angle) / 2,
                center[1] + dy + width * np.cos(angle) / 2,
            ],
            [
                center[0] + dx + width * np.sin(angle) / 2,
                center[1] + dy - width * np.cos(angle) / 2,
            ],
            [
                center[0] - dx + width * np.sin(angle) / 2,
                center[1] - dy - width * np.cos(angle) / 2,
            ],
        ]
    )
    polygon = Polygon(
        corners, closed=True, edgecolor="b" if side == "left" else "g", fill=False
    )
    ax.add_patch(polygon)


def draw_footsteps(
    path,
    foot_steps,
    foot_size,
    y_offset_com_to_foot,
    fig_size=(10, 6),
    title=None,
    x_label=None,
    y_label=None,
    save_config=False,
    save_path=None,
    time_suffix=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect("equal")

    def plot():
        ax.plot(path[:, 0], path[:, 1], "r-", label="Cubic Hermite Path")

        # Draw each footstep
        for step in foot_steps:
            if step.support_leg == "both":
                dx = -y_offset_com_to_foot * np.sin(step.position[2])
                dy = y_offset_com_to_foot * np.cos(step.position[2])

                left_foot_pos = [step.position[0] + dx, step.position[1] + dy]
                draw_foot(ax, left_foot_pos, foot_size, step.position[2], "left")
                right_foot_pos = [step.position[0] - dx, step.position[1] - dy]
                draw_foot(ax, right_foot_pos, foot_size, step.position[2], "right")
            else:
                draw_foot(
                    ax, step.position[:2], foot_size, step.position[2], step.support_leg
                )

    vis_function = make_vis_function(
        plot,
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        save_config=save_config,
        save_path=save_path,
        time_suffix=time_suffix,
    )
    return vis_function
