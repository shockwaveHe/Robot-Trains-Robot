import argparse
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Polygon

from toddlerbot.visualization.vis_utils import (
    load_and_run_visualization,
    make_vis_function,
)

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "D", "v", "^", "<"]
COLORS = ["b", "g", "r", "c", "y", "k"]


def plot_waist_mapping(
    joint_limits: Dict[str, List[float]],
    waist_ik: Callable[..., List[float]],
    save_path: str,
    file_name: str = "waist_mapping",
):
    # Prepare data for plotting
    roll_limits = joint_limits["waist_roll"]
    yaw_limits = joint_limits["waist_yaw"]
    act_1_limits = joint_limits["waist_act_1"]
    act_2_limits = joint_limits["waist_act_2"]

    step_rad = 0.02
    tol = 1e-3
    roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)  # type: ignore
    yaw_range = np.arange(yaw_limits[0], yaw_limits[1] + step_rad, step_rad)  # type: ignore
    roll_grid, yaw_grid = np.meshgrid(roll_range, yaw_range, indexing="ij")  # type: ignore

    act_1_grid = np.zeros_like(roll_grid)
    act_2_grid = np.zeros_like(yaw_grid)
    for i in range(len(roll_range)):  # type: ignore
        for j in range(len(yaw_range)):  # type: ignore
            act_pos: List[float] = waist_ik([roll_range[i], yaw_range[j]])
            act_1_grid[i, j] = act_pos[0]
            act_2_grid[i, j] = act_pos[1]

    valid_mask = (
        (act_1_grid >= act_1_limits[0] - tol)
        & (act_1_grid <= act_1_limits[1] + tol)
        & (act_2_grid >= act_2_limits[0] - tol)
        & (act_2_grid <= act_2_limits[1] + tol)
    )

    # Create a color array based on the valid_mask
    colors = np.where(valid_mask.flatten(), "red", "white")  # type: ignore

    n_rows = 1
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # type: ignore

    for i, ax in enumerate(axs.flat):  # type: ignore
        if i == 0:
            plot_scatter_graph(
                act_2_grid.flatten(),
                act_1_grid.flatten(),
                colors,
                x_label="Actuator 1 (rad)",
                y_label="Actuator 2 (rad)",
                title="Waist Forward Mapping",
                ax=ax,
            )()
        else:
            plot_scatter_graph(
                yaw_grid.flatten(),
                roll_grid.flatten(),
                colors,
                x_label="Roll (rad)",
                y_label="Yaw (rad)",
                title="Waist Inverse Mapping",
                save_config=True,
                save_path=save_path,
                file_name=file_name,
                ax=ax,
            )()


def plot_ankle_mapping(
    joint_limits: Dict[str, List[float]],
    ankle_ik: Callable[..., List[float]],
    save_path: str,
    file_name: str = "ankle_mapping",
):
    # Prepare data for plotting
    roll_limits = joint_limits["left_ank_roll"]
    pitch_limits = joint_limits["left_ank_pitch"]
    act_1_limits = joint_limits["left_ank_act_1"]
    act_2_limits = joint_limits["left_ank_act_2"]

    step_rad = 0.02
    tol = 1e-3
    roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)  # type: ignore
    pitch_range = np.arange(pitch_limits[0], pitch_limits[1] + step_rad, step_rad)  # type: ignore
    roll_grid, pitch_grid = np.meshgrid(roll_range, pitch_range, indexing="ij")  # type: ignore

    act_1_grid = np.zeros_like(roll_grid)
    act_2_grid = np.zeros_like(pitch_grid)
    for i in range(len(roll_range)):  # type: ignore
        for j in range(len(pitch_range)):  # type: ignore
            act_pos: List[float] = ankle_ik([roll_range[i], pitch_range[j]])
            act_1_grid[i, j] = act_pos[0]
            act_2_grid[i, j] = act_pos[1]

    valid_mask = (
        (act_1_grid >= act_1_limits[0] - tol)
        & (act_1_grid <= act_1_limits[1] + tol)
        & (act_2_grid >= act_2_limits[0] - tol)
        & (act_2_grid <= act_2_limits[1] + tol)
    )

    # Create a color array based on the valid_mask
    colors = np.where(valid_mask.flatten(), "red", "white")  # type: ignore

    n_rows = 1
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # type: ignore

    for i, ax in enumerate(axs.flat):  # type: ignore
        if i == 0:
            plot_scatter_graph(
                act_2_grid.flatten(),
                act_1_grid.flatten(),
                colors,
                x_label="Actuator 1 (rad)",
                y_label="Actuator 2 (rad)",
                title="Ankle Forward Mapping",
                ax=ax,
            )()
        else:
            plot_scatter_graph(
                pitch_grid.flatten(),
                roll_grid.flatten(),
                colors,
                x_label="Roll (rad)",
                y_label="Pitch (rad)",
                title="Ankle Inverse Mapping",
                save_config=True,
                save_path=save_path,
                file_name=file_name,
                ax=ax,
            )()


def plot_one_footstep(
    ax: plt.Axes,
    center: npt.NDArray[np.float32],
    size: Tuple[float, float],
    angle: float,
    side: int,
) -> None:
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
        corners,  # type: ignore
        closed=True,
        edgecolor="b" if side == 0 else "g",
        fill=False,
    )
    ax.add_patch(polygon)


def plot_footsteps(
    path: npt.NDArray[np.float32],
    foot_pos_list: npt.NDArray[np.float32],
    support_leg_list: List[int],
    foot_size: Tuple[float, float],
    foot_to_com_y: float,
    fig_size: Tuple[int, int] = (10, 6),
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_config: bool = False,
    save_path: str = "",
    file_name: str = "",
    file_suffix: str = "",
    ax: Any = None,
) -> Callable[[], None]:
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)  # type: ignore
        ax.set_aspect("equal")

    def plot() -> None:
        ax.plot(path[:, 0], path[:, 1], "r-", label="Cubic Hermite Path")  # type: ignore

        # Draw each footstep
        for foot_pos, support_leg in zip(foot_pos_list, support_leg_list):
            if support_leg == 2:
                dx = -foot_to_com_y * np.sin(foot_pos[2])
                dy = foot_to_com_y * np.cos(foot_pos[2])

                left_foot_pos = [foot_pos[0] + dx, foot_pos[1] + dy]
                plot_one_footstep(
                    ax, np.array(left_foot_pos), foot_size, foot_pos[2], 0
                )
                right_foot_pos = [foot_pos[0] - dx, foot_pos[1] - dy]
                plot_one_footstep(
                    ax, np.array(right_foot_pos), foot_size, foot_pos[2], 1
                )
            else:
                plot_one_footstep(ax, foot_pos[:2], foot_size, foot_pos[2], support_leg)

    vis_function: Any = make_vis_function(
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


def plot_loop_time(
    loop_time_dict: Dict[str, List[float]], save_path: str, file_name: str = "loop_time"
):
    plot_line_graph(
        list(loop_time_dict.values()),
        legend_labels=list(loop_time_dict.keys()),
        title="Loop Time",
        x_label="Iterations",
        y_label="Time (ms)",
        save_config=True,
        save_path=save_path,
        file_name=file_name,
    )()


def plot_joint_angle_tracking(
    time_seq_dict: Dict[str, List[float]],
    time_seq_ref_dict: Dict[str, List[float]],
    joint_angle_dict: Dict[str, List[float]],
    joint_angle_ref_dict: Dict[str, List[float]],
    joint_limits: Dict[str, List[float]],
    save_path: str,
    file_name: str = "joint_angle_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
    set_ylim: bool = True,
    line_suffix: List[str] = ["_obs", "_act"],
):
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    joint_name_list: List[str] = []
    legend_labels: List[str] = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        x_list.append(time_seq_ref_dict[name])
        y_list.append(joint_angle_dict[name])
        y_list.append(joint_angle_ref_dict[name])
        joint_name_list.append(name)
        legend_labels.append(name + line_suffix[0])
        legend_labels.append(name + line_suffix[1])

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # type: ignore
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore

    for i, ax in enumerate(axs.flat):  # type: ignore
        if i >= n_plots:
            ax.set_visible(False)  # type: ignore
            continue

        if set_ylim:
            y_min, y_max = joint_limits[joint_name_list[i]]
            ax.set_ylim(y_min - 0.1, y_max + 0.1)  # type: ignore

        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{joint_name_list[i]}" if len(title_list) == 0 else title_list[i],
            x_label="Time (s)",
            y_label="Position (rad)",
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else "",
            file_name=file_name if i == n_plots - 1 else "",
            file_suffix=file_suffix,
            ax=ax,  # type: ignore
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()


def plot_joint_velocity_tracking(
    time_seq_dict: Dict[str, List[float]],
    joint_vel_dict: Dict[str, List[float]],
    save_path: str,
    file_name: str = "joint_velocity_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
):
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    legend_labels: List[str] = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        y_list.append(joint_vel_dict[name])
        legend_labels.append(name)

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # type: ignore
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore

    for i, ax in enumerate(axs.flat):  # type: ignore
        if i >= n_plots:
            ax.set_visible(False)  # type: ignore
            continue

        ax.set_ylim(-5, 5)  # type: ignore

        plot_line_graph(
            y_list[i],
            x_list[i],
            title=f"{legend_labels[i]}" if len(title_list) == 0 else title_list[i],
            x_label="Time (s)",
            y_label="Velocity (rad/s)",
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else "",
            file_name=file_name if i == n_plots - 1 else "",
            file_suffix=file_suffix,
            ax=ax,  # type: ignore
            legend_labels=[legend_labels[i]],
        )()


def plot_orientation_tracking(
    time_list: List[float],
    euler_list: List[npt.NDArray[np.float32]],
    save_path: str,
    file_name: str = "euler_angles_tracking",
    file_suffix: str = "",
):
    plot_line_graph(
        np.array(euler_list).T,
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
    time_list: List[float],
    ang_vel_list: List[npt.NDArray[np.float32]],
    save_path: str,
    file_name: str = "angular_velocity_tracking",
    file_suffix: str = "",
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
    y: Any,
    x: Any = None,
    fig_size: Tuple[int, int] = (10, 6),
    legend_labels: List[str] = [],
    line_styles: List[str] = [],
    line_colors: List[str] = [],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_config: bool = False,
    save_path: str = "",
    file_name: str = "",
    file_suffix: str = "",
    ax: Any = None,
    checkpoint_period: List[int] = [],
):
    if ax is None:
        plt.figure(figsize=fig_size)  # type: ignore
        ax = plt.gca()  # type: ignore

    def plot():
        # Ensure line_styles and line_colors are lists and have sufficient length
        line_styles_local = line_styles if len(line_styles) > 0 else LINE_STYLES
        line_colors_local = line_colors if len(line_colors) > 0 else COLORS

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
                            ax.plot(xi[idx], value, MARKERS[i], color=color)  # type: ignore
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
                            MARKERS[0],
                            color=line_colors_local[0],
                        )

        if legend_labels:
            ax.legend()

    # Create and return a visualization function using the make_vis_function
    vis_function: Any = make_vis_function(
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


def plot_scatter_graph(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    colors: npt.NDArray[np.float32],
    fig_size: Tuple[int, int] = (10, 6),
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    legend_label: str = "",
    save_config: bool = False,
    save_path: str = "",
    file_name: str = "",
    file_suffix: str = "",
    ax: Any = None,
):
    if ax is None:
        plt.figure(figsize=fig_size)  # type: ignore
        ax = plt.gca()  # type: ignore

    def plot():
        # Ensure point_styles and point_colors are lists and have sufficient length
        ax.scatter(
            x,
            y,
            color=colors,
            label=legend_label if len(legend_label) > 0 else None,
        )

        if len(legend_label) > 0:
            ax.legend()

    # Create and return a visualization function using the make_vis_function
    vis_function: Any = make_vis_function(
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
