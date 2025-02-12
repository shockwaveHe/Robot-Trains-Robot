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

# This script contains all sorts of visualization functions.

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "D", "v", "^", "<"]
COLORS = ["b", "g", "r", "c", "y", "k"]


def plot_teleop_dataset(
    action_data: npt.NDArray[np.float32],
    episode_ends: np.ndarray,
    save_path: str,
    file_name: str,
    file_suffix: str = "",
):
    episode_starts = np.concatenate(([0], episode_ends[:-1]))

    motor_traj_list = []
    for e_idx in range(len(episode_ends)):
        start_idx = episode_starts[e_idx]
        end_idx = episode_ends[e_idx]
        # Extract the joint trajectory for this episode
        motor_traj_list.append(action_data[start_idx:end_idx])

    # Compute the 75th percentile of episode lengths
    episode_lengths = [traj.shape[0] for traj in motor_traj_list]
    max_length = int(np.percentile(episode_lengths, 75))

    # Create a figure and 4x4 subplots for the 16 joints
    fig, axs = plt.subplots(4, 4, figsize=(15, 10))
    axs = axs.flatten()  # Flatten so we can iterate through them easily
    num_joints = action_data.shape[-1]
    # For each joint, we will plot all episodes
    for joint_idx in range(num_joints):
        ax = axs[joint_idx]

        # Plot each episode as a separate line
        plot_line_graph(
            [
                motor_traj_list[i][:max_length, joint_idx]
                for i in range(len(motor_traj_list))
            ],
            title=f"Joint {joint_idx}",
            x_label="Time Step",
            y_label="Position",
            save_config=True if joint_idx == num_joints - 1 else False,
            save_path=save_path if joint_idx == num_joints - 1 else "",
            file_name=file_name if joint_idx == num_joints - 1 else "",
            file_suffix=file_suffix,
            ax=ax,
        )()


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
    roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)
    yaw_range = np.arange(yaw_limits[0], yaw_limits[1] + step_rad, step_rad)
    roll_grid, yaw_grid = np.meshgrid(roll_range, yaw_range, indexing="ij")

    act_1_grid = np.zeros_like(roll_grid)
    act_2_grid = np.zeros_like(yaw_grid)
    for i in range(len(roll_range)):
        for j in range(len(yaw_range)):
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
    colors = np.where(valid_mask.flatten(), "red", "white")

    n_rows = 1
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

    for i, ax in enumerate(axs.flat):
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
    roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)
    pitch_range = np.arange(pitch_limits[0], pitch_limits[1] + step_rad, step_rad)
    roll_grid, pitch_grid = np.meshgrid(roll_range, pitch_range, indexing="ij")

    act_1_grid = np.zeros_like(roll_grid)
    act_2_grid = np.zeros_like(pitch_grid)
    for i in range(len(roll_range)):
        for j in range(len(pitch_range)):
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
    colors = np.where(valid_mask.flatten(), "red", "white")

    n_rows = 1
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

    for i, ax in enumerate(axs.flat):
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


def plot_motor_vel_tor_mapping(
    motor_vel_list: List[float],
    motor_tor_list: List[float],
    save_path: str,
    file_name: str = "motor_vel_tor_mapping",
):
    plot_scatter_graph(
        np.abs(motor_vel_list, dtype=np.float32),
        np.abs(motor_tor_list, dtype=np.float32),
        "blue",
        x_label="Torque (Nm) or Current (mA)",
        y_label="Velocity (rad/s)",
        title="Motor Velocity-Torque Mapping",
        save_config=True,
        save_path=save_path,
        file_name=file_name,
    )()


def plot_one_footstep(
    ax: plt.Axes,
    center: npt.NDArray[np.float32],
    size: Tuple[float, float],
    angle: float,
    side: int,
):
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
        corners,
        closed=True,
        edgecolor="b" if side == 0 else "g",
        fill=False,
    )
    ax.add_patch(polygon)

    return corners


def plot_footsteps(
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
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect("equal")

    def plot():
        # Draw each footstep
        all_x = []
        all_y = []
        for foot_pos, support_leg in zip(foot_pos_list, support_leg_list):
            if support_leg == 2:
                dx = -foot_to_com_y * np.sin(foot_pos[2])
                dy = foot_to_com_y * np.cos(foot_pos[2])

                left_foot_pos = [foot_pos[0] + dx, foot_pos[1] + dy]
                corners_left = plot_one_footstep(
                    ax, np.array(left_foot_pos), foot_size, foot_pos[2], 0
                )
                right_foot_pos = [foot_pos[0] - dx, foot_pos[1] - dy]
                corners_right = plot_one_footstep(
                    ax, np.array(right_foot_pos), foot_size, foot_pos[2], 1
                )
                all_x.extend(corners_left[:, 0])
                all_x.extend(corners_right[:, 0])
                all_y.extend(corners_left[:, 1])
                all_y.extend(corners_right[:, 1])
            else:
                corners = plot_one_footstep(
                    ax, foot_pos[:2], foot_size, foot_pos[2], support_leg
                )
                all_x.extend(corners[:, 0])
                all_y.extend(corners[:, 1])

        padding = 0.05  # Add some padding around the footsteps
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

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


def plot_path_tracking(
    time_obs_list: List[float],
    pos_obs_list: List[npt.NDArray[np.float32]],
    euler_obs_list: List[npt.NDArray[np.float32]],
    control_inputs_dict: Dict[str, List[float]],
    save_path: str,
    file_name: str = "path_tracking",
):
    """
    Plots the observed path (pos_obs_list) and the integrated path of walk commands.
    """
    # Extract observed positions (torso)
    obs_pos = np.array(pos_obs_list)

    # Extract walk commands
    walk_command_list = [
        control_inputs_dict["walk_x"],
        control_inputs_dict["walk_y"],
    ]
    turn_command_list = control_inputs_dict["walk_turn"]  # Extract turn commands

    if len(walk_command_list) == 0 or len(turn_command_list) == 0:
        raise ValueError("No walk or turn commands found in control_inputs_dict.")

    # Assume walk commands are in the format [[vx1, vy1], [vx2, vy2], ...]
    walk_commands = np.array(walk_command_list).T
    turn_commands = np.array(turn_command_list)  # Angular velocity (yaw rate)

    dt = np.diff(time_obs_list)  # Compute time intervals
    dt = np.append(dt, dt[-1])

    # Integrate velocities to compute positions
    target_pos = np.tile(obs_pos[0], (len(walk_commands), 1))  # [x, y]
    target_orientation = 0.0  # Start with 0 yaw (global frame)

    for i in range(1, len(walk_commands)):
        if walk_commands[i, 0] < -0.9:
            walk_commands[i, 0] = 0.1
        elif walk_commands[i, 0] > 0.9:
            walk_commands[i, 0] = -0.05
        else:
            walk_commands[i, 0] = 0

        if walk_commands[i, 1] < -0.9:
            walk_commands[i, 1] = 0.05
        elif walk_commands[i, 1] > 0.9:
            walk_commands[i, 1] = -0.05
        else:
            walk_commands[i, 1] = 0

        if turn_commands[i] < -0.9:
            turn_commands[i] = 0.25
        elif turn_commands[i] > 0.9:
            turn_commands[i] = -0.25

        target_orientation += turn_commands[i] * dt[i]
        # Compute the rotation matrix from the current orientation
        rotation_matrix = np.array(
            [
                [np.cos(target_orientation), -np.sin(target_orientation)],
                [np.sin(target_orientation), np.cos(target_orientation)],
            ]
        )

        # Transform the walking command to the global frame
        global_walk_command = rotation_matrix @ walk_commands[i]

        target_pos[i, 0] = target_pos[i - 1, 0] + global_walk_command[0] * dt[i]
        target_pos[i, 1] = target_pos[i - 1, 1] + global_walk_command[1] * dt[i]

    plot_line_graph(
        [obs_pos[:, 1], target_pos[:, 1]],
        [obs_pos[:, 0], target_pos[:, 0]],
        legend_labels=["Observed", "Target"],
        title="Path Tracking",
        x_label="X (m)",
        y_label="Y (m)",
        save_config=True,
        save_path=save_path,
        file_name=file_name,
    )()


def plot_joint_tracking(
    time_seq_dict: Dict[str, List[float]],
    time_seq_ref_dict: Dict[str, List[float]],
    joint_data_dict: Dict[str, List[float]],
    joint_data_ref_dict: Dict[str, List[float]],
    joint_limits: Dict[str, List[float]],
    save_path: str,
    x_label: str = "Time (s)",
    y_label: str = "Position (rad)",
    file_name: str = "motor_pos_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
    set_ylim: bool = False,
    line_suffix: List[str] = ["_obs", "_act"],
):
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    joint_name_list: List[str] = []
    legend_labels: List[str] = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        x_list.append(time_seq_ref_dict[name])
        y_list.append(joint_data_dict[name])
        y_list.append(joint_data_ref_dict[name])
        joint_name_list.append(name)
        legend_labels.append(name + line_suffix[0])
        legend_labels.append(name + line_suffix[1])

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axs.flat):
        if i >= n_plots:
            ax.set_visible(False)
            continue

        if set_ylim:
            y_min, y_max = joint_limits[joint_name_list[i]]
            ax.set_ylim(y_min - 0.1, y_max + 0.1)

        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{joint_name_list[i]}" if len(title_list) == 0 else title_list[i],
            x_label=x_label,
            y_label=y_label,
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else "",
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()


def plot_joint_tracking_frequency(
    time_seq_dict: Dict[str, List[float]],
    time_seq_ref_dict: Dict[str, List[float]],
    joint_data_dict: Dict[str, List[float]],
    joint_data_ref_dict: Dict[str, List[float]],
    save_path: str,
    x_label: str = "Frequency (Hz)",
    y_label: str = "Magnitude",
    file_name: str = "motor_freq_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
    set_ylim: bool = False,
    line_suffix: List[str] = ["_obs", "_ref"],
):
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    joint_name_list: List[str] = []
    legend_labels: List[str] = []

    # For each joint, compute the FFT of the joint data and the reference joint data
    for name in time_seq_dict.keys():
        joint_data = joint_data_dict[name]
        time_seq_ref = time_seq_ref_dict[name]
        joint_data_ref = joint_data_ref_dict[name]

        # Calculate the time step (assuming uniform sampling)
        time_step = np.mean(np.diff(time_seq_ref))

        # FFT (Fourier Transform) of the joint position data and reference data
        joint_data_fft = np.fft.fft(joint_data)
        freqs = np.fft.fftfreq(len(joint_data), time_step)
        joint_data_ref_fft = np.fft.fft(joint_data_ref)
        freqs_ref = np.fft.fftfreq(len(joint_data_ref), time_step)

        # Use only the positive frequencies
        pos_freqs = freqs[: len(freqs) // 2]
        pos_freqs_ref = freqs_ref[: len(freqs_ref) // 2]
        pos_magnitudes = np.abs(joint_data_fft[: len(joint_data_fft) // 2])
        pos_magnitudes_ref = np.abs(joint_data_ref_fft[: len(joint_data_ref_fft) // 2])

        x_list.append(list(pos_freqs))
        x_list.append(list(pos_freqs_ref))
        y_list.append(list(pos_magnitudes))
        y_list.append(list(pos_magnitudes_ref))

        joint_name_list.append(name)
        legend_labels.append(name + line_suffix[0])  # Observation label
        legend_labels.append(name + line_suffix[1])  # Reference label

    n_plots = len(time_seq_dict)
    n_rows = int(np.ceil(n_plots / 3))
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axs.flat):
        if i >= n_plots:
            ax.set_visible(False)
            continue

        if set_ylim:
            ax.set_ylim(-0.1, 100)

        ax.set_yscale("log")

        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{joint_name_list[i]}" if len(title_list) == 0 else title_list[i],
            x_label=x_label,
            y_label=y_label,
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else "",
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()


def plot_joint_tracking_single(
    time_seq_dict: Dict[str, List[float]],
    joint_data_dict: Dict[str, List[float]],
    save_path: str,
    x_label: str = "Time (s)",
    y_label: str = "Velocity (rad/s)",
    file_name: str = "motor_vel_tracking",
    file_suffix: str = "",
    title_list: List[str] = [],
    set_ylim: bool = False,
):
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    legend_labels: List[str] = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        y_list.append(joint_data_dict[name])
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

        if set_ylim:
            ax.set_ylim(-5, 5)

        plot_line_graph(
            y_list[i],
            x_list[i],
            title=f"{legend_labels[i]}" if len(title_list) == 0 else title_list[i],
            x_label=x_label,
            y_label=y_label,
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else "",
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=[legend_labels[i]],
        )()


def plot_sim2real_gap_line(
    time_sim_list: List[float],
    time_real_list: List[float],
    data_sim: npt.NDArray[np.float32],
    data_real: npt.NDArray[np.float32],
    save_path: str,
    title: str = "Euler Angles",
    x_label: str = "Time (s)",
    y_label: str = "Euler Angles (rad)",
    axis_names: List[str] = ["roll", "pitch", "yaw"],
    file_name: str = "euler_gap",
    file_suffix: str = "",
):
    for data_sim, angle_real, axis_name in zip(data_sim.T, data_real.T, axis_names):
        plot_line_graph(
            [data_sim, angle_real],
            [
                time_sim_list,
                time_real_list,
            ],
            legend_labels=[
                f"{axis_name}_sim",
                f"{axis_name}_real",
            ],
            title=title,
            x_label=x_label,
            y_label=y_label,
            save_config=True,
            save_path=save_path,
            file_name=f"{file_name}_{axis_name}",
            file_suffix=file_suffix,
        )()


def plot_sim2real_gap_bar(
    rmse_dict: Dict[str, float],
    rmse_label: str,
    save_path: str,
    title: str = "Root Mean Squared Error by Joint",
    x_label: str = "Joints",
    y_label: str = "Root Mean Squared Error",
    file_name: str = "sim2real_gap",
    file_suffix: str = "",
):
    joint_labels = list(rmse_dict.keys())

    # Call the plot_bar_graph function
    plot_bar_graph(
        y=list(rmse_dict.values()),
        x=np.arange(len(rmse_dict)),
        fig_size=(int(len(rmse_dict) / 3), 6),
        legend_labels=[rmse_label],
        title=title,
        x_label=x_label,
        y_label=y_label,
        bar_colors=["b"],
        bar_width=0.25,
        save_config=True,
        save_path=save_path,
        file_name=file_name,
        file_suffix=file_suffix,
        joint_labels=joint_labels,  # Pass the joint labels
    )()


def plot_bar_graph(
    y: Any,
    x: Any = None,
    fig_size: Tuple[int, int] = (10, 6),
    legend_labels: List[str] = [],
    bar_colors: List[str] = [],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_config: bool = False,
    save_path: str = "",
    file_name: str = "",
    file_suffix: str = "",
    ax: Any = None,
    bar_width: float = 0.25,
    joint_labels: List[str] = [],  # New parameter for joint labels
    number_font_size: int = 0,
):
    if ax is None:
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    def plot():
        # Ensure bar_colors are lists and have sufficient length
        bar_colors_local = bar_colors if len(bar_colors) > 0 else COLORS

        # Determine if x is None and set it to the index of y if so
        if x is None:
            x_local = (
                np.arange(len(y[0])) if isinstance(y[0], list) else np.arange(len(y))
            )
        else:
            x_local = x

        # Add number labels on each bar
        def add_number_labels(bars: List[Any]):
            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    f"{yval:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=number_font_size,
                )

        # Plotting multiple bars if y is a list of lists
        if isinstance(y[0], list) or isinstance(y[0], np.ndarray):  # Multiple groups
            for i, sub_y in enumerate(y):
                bar_positions = x_local + i * bar_width
                color = bar_colors_local[i % len(bar_colors_local)]
                bars = ax.bar(
                    bar_positions,
                    sub_y,
                    width=bar_width,
                    color=color,
                    label=legend_labels[i] if legend_labels else None,
                )

                if number_font_size > 0:
                    add_number_labels(bars)

        else:  # Single group of bars
            bars = ax.bar(
                x_local,
                y,
                width=bar_width,
                color=bar_colors_local[0],
                label=legend_labels[0] if legend_labels else None,
            )

            if number_font_size > 0:
                add_number_labels(bars)

        # Set joint labels as x-tick labels
        if joint_labels:
            ax.set_xticks(x_local + bar_width)  # Adjusting for center alignment
            ax.set_xticklabels(joint_labels, rotation=90, ha="right")

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
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    def plot():
        # Ensure line_styles and line_colors are lists and have sufficient length
        line_styles_local = line_styles if len(line_styles) > 0 else LINE_STYLES
        line_colors_local = line_colors if len(line_colors) > 0 else COLORS

        # Determine if x is None and set it to the index of y if so
        if x is None:
            x_local = (
                [list(range(len(sub_y))) for sub_y in y]
                if isinstance(y[0], list) or isinstance(y[0], np.ndarray)
                else list(range(len(y)))
            )
        else:
            x_local = x

        if isinstance(y[0], list) or isinstance(y[0], np.ndarray):  # Multiple lines
            for i, sub_y in enumerate(y):
                xi = (
                    x_local[i]
                    if isinstance(x_local[0], list)
                    or isinstance(x_local[0], np.ndarray)
                    else x_local
                )
                style = line_styles_local[i % len(line_styles_local)]
                color = line_colors_local[i % len(line_colors_local)]
                ax.plot(
                    xi,
                    sub_y,
                    style,
                    color=color,
                    alpha=0.7,
                    label=legend_labels[i] if legend_labels else None,
                    linewidth=0.5,
                )

                if checkpoint_period and checkpoint_period[i]:
                    for idx, value in enumerate(sub_y):
                        if idx % checkpoint_period[i] == 0:
                            ax.plot(xi[idx], value, MARKERS[i], color=color)
        else:  # Single line
            ax.plot(
                x_local,
                y,
                line_styles_local[0],
                color=line_colors_local[0],
                alpha=0.7,
                label=legend_labels[0] if legend_labels else None,
                linewidth=0.5,
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
    colors: npt.NDArray[np.float32] | str,
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
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    def plot():
        # Ensure point_styles and point_colors are lists and have sufficient length
        ax.scatter(
            x,
            y,
            s=1.0,
            c=colors,
            alpha=0.7,
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
