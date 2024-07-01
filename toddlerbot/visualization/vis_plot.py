import argparse
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from toddlerbot.visualization.vis_utils import (
    load_and_run_visualization,
    make_vis_function,
)

LINE_STYLES = ["-", "--", "-.", ":"]
LINE_COLORS = ["b", "g", "r", "c", "y", "k"]
LINE_MARKERS = ["o", "s", "D", "v", "^", "<"]


def plot_ankle_mapping(self, ankle_ik):
    pitch_range = np.linspace(-np.pi / 2, np.pi / 2, 180)
    roll_range = np.linspace(-np.pi / 2, np.pi / 2, 180)
    pitch_grid, roll_grid = np.meshgrid(pitch_range, roll_range)
    d1_values = np.zeros_like(pitch_grid)
    d2_values = np.zeros_like(roll_grid)

    for i in range(pitch_grid.shape[0]):
        for j in range(pitch_grid.shape[1]):
            d1, d2 = ankle_ik((pitch_grid[i, j], roll_grid[i, j]))
            d1_values[i, j] = d1
            d2_values[i, j] = d2

    valid_mask_d1 = (d1_values >= 0) & (d1_values <= 4095)
    valid_mask_d2 = (d2_values >= 0) & (d2_values <= 4095)
    valid_both = valid_mask_d1 & valid_mask_d2

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contour(
        pitch_grid, roll_grid, valid_both, levels=[0.5], colors="red", linewidths=2
    )

    ax.set_title("Ankle Position Validity Mapping")
    ax.set_xlabel("Ankle Pitch (radians)")
    ax.set_ylabel("Ankle Roll (radians)")

    plt.show()


def plot_one_footstep(ax, center, size, angle, side):
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


def plot_footsteps(
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
    file_suffix=None,
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
                plot_one_footstep(
                    ax, left_foot_pos, foot_size, step.position[2], "left"
                )
                right_foot_pos = [step.position[0] - dx, step.position[1] - dy]
                plot_one_footstep(
                    ax, right_foot_pos, foot_size, step.position[2], "right"
                )
            else:
                plot_one_footstep(
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
        file_suffix=file_suffix,
    )
    return vis_function


def plot_joint_angle_tracking(
    time_seq_dict: Dict[str, List[float]],
    time_seq_ref: Dict[str, List[float]],
    joint_angle_dict: Dict[str, List[float]],
    joint_angle_ref_dict: Dict[str, List[float]],
    save_path: str,
    file_name: str = "joint_angle_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
):
    # all_angles = np.concatenate(
    #     list(joint_angle_dict.values()) + list(joint_angle_ref_dict.values())
    # )
    # global_ymin, global_ymax = np.min(all_angles) - 0.1, np.max(all_angles) + 0.1

    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    legend_labels: List[str] = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        if isinstance(time_seq_ref, list):
            x_list.append(time_seq_ref)
        else:
            x_list.append(time_seq_ref[name])
        y_list.append(joint_angle_dict[name])
        y_list.append(joint_angle_ref_dict[name])
        legend_labels.append(name)
        legend_labels.append(name + "_ref")

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # type: ignore
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore

    for i, ax in enumerate(axs.flat):  # type: ignore
        if i >= n_plots:
            ax.set_visible(False)  # type: ignore
            continue

        # ax.set_ylim(global_ymin, global_ymax)
        # ax.set_facecolor(colors_dict[motor_params[legend_labels[2 * i]].brand])

        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{legend_labels[2*i]}" if len(title_list) == 0 else title_list[i],
            x_label="Time (s)",
            y_label="Position (rad)",
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else None,
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()


def plot_joint_velocity_tracking(
    time_seq_dict,
    joint_vel_dict,
    save_path,
    file_name="joint_velocity_tracking",
    file_suffix="",
    title_list=None,
    motor_params=None,
    colors_dict=None,
):
    # all_angles = np.concatenate(
    #     list(joint_angle_dict.values()) + list(joint_angle_ref_dict.values())
    # )
    # global_ymin, global_ymax = np.min(all_angles) - 0.1, np.max(all_angles) + 0.1

    def get_brand(joint_name):
        if joint_name in motor_params:
            return motor_params[joint_name].brand
        return "default"

    time_seq_dict = {
        joint_name: x
        for joint_name, x in sorted(
            time_seq_dict.items(), key=lambda item: (get_brand(item[0]), item[0])
        )
    }
    joint_vel_dict = {
        joint_name: x
        for joint_name, x in sorted(
            joint_vel_dict.items(), key=lambda item: (get_brand(item[0]), item[0])
        )
    }

    x_list = []
    y_list = []
    legend_labels = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        y_list.append(joint_vel_dict[name])
        legend_labels.append(name)

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axs.flat):
        if i >= n_plots:
            ax.set_visible(False)
            continue

        # ax.set_ylim(global_ymin, global_ymax)
        if motor_params is not None and colors_dict is not None:
            ax.set_facecolor(colors_dict[motor_params[legend_labels[i]].brand])

        plot_line_graph(
            y_list[i],
            x_list[i],
            title=f"{legend_labels[i]}" if title_list is None else title_list[i],
            x_label="Time (s)",
            y_label="Velocity (rad/s)",
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else None,
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=legend_labels[i],
        )()


def plot_orientation_tracking(
    time_list,
    euler_angle_list,
    save_path,
    file_name="euler_angles_tracking",
    file_suffix="",
):
    plot_line_graph(
        np.array(euler_angle_list).T,
        time_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Euler Angles Over Time",
        x_label="Time Step",
        y_label="Euler Angles (rad)",
        save_config=True,
        save_path=save_path,
        file_name=file_name,
        file_suffix=file_suffix,
    )()


def plot_angular_velocity_tracking(
    time_list,
    ang_vel_list,
    save_path,
    file_name="angular_velocity_tracking",
    file_suffix="",
):
    plot_line_graph(
        np.array(ang_vel_list).T,
        time_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Angular Velocities Over Time",
        x_label="Time Step",
        y_label="Angular Velocity (rad/s)",
        save_config=True,
        save_path=save_path,
        file_name=file_name,
        file_suffix=file_suffix,
    )()


def plot_line_graph(
    y,
    x=None,
    fig_size=(10, 6),
    legend_labels=None,
    line_styles=None,
    line_colors=None,
    title=None,
    x_label=None,
    y_label=None,
    save_config=False,
    save_path=None,
    file_name=None,
    file_suffix=None,
    ax=None,
    checkpoint_period=None,
):
    if ax is None:
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    def plot():
        # Ensure line_styles and line_colors are lists and have sufficient length
        line_styles_local = line_styles if line_styles is not None else LINE_STYLES
        line_colors_local = line_colors if line_colors is not None else LINE_COLORS

        # Determine if x is None and set it to the index of y if so
        if x is None:
            x_local = (
                list(range(len(y)))
                if not isinstance(y[0], list)
                else [list(range(len(sub_y))) for sub_y in y]
            )
        else:
            x_local = x

        if isinstance(y[0], list) or isinstance(y[0], np.ndarray):  # Multiple lines
            for i, sub_y in enumerate(y):
                xi = x_local[i] if isinstance(x_local[0], list) else x_local
                style = line_styles_local[i % len(line_styles_local)]
                color = line_colors_local[i % len(line_colors_local)]
                ax.plot(
                    xi,
                    sub_y,
                    style,
                    color=color,
                    label=legend_labels[i] if legend_labels else None,
                )

                if checkpoint_period and checkpoint_period[i]:
                    for idx, value in enumerate(sub_y):
                        if idx % checkpoint_period[i] == 0:
                            ax.plot(xi[idx], value, LINE_MARKERS[i], color=color)
        else:  # Single line
            ax.plot(
                x_local,
                y,
                line_styles_local[0],
                color=line_colors_local[0],
                label=legend_labels[0] if legend_labels else None,
            )

            if checkpoint_period and checkpoint_period[0]:
                for idx, value in enumerate(y):
                    if idx % checkpoint_period[0] == 0:
                        ax.plot(
                            x_local[idx],
                            value,
                            LINE_MARKERS[0],
                            color=line_colors_local[0],
                        )

        if legend_labels:
            ax.legend()

    # Create and return a visualization function using the make_vis_function
    vis_function = make_vis_function(
        plot,
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        save_config=save_config,
        save_path=save_path,
        file_name=file_name,
        file_suffix=file_suffix,
    )

    return vis_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a visualization function specified in a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    load_and_run_visualization(args.config)
