import argparse

import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.utils.vis_utils import load_and_run_visualization, make_vis_function

LINE_STYLES = ["-", "--", "-.", ":"]
LINE_COLORS = ["b", "g", "r", "c", "y", "k"]
LINE_MARKERS = ["o", "s", "D", "v", "^", "<"]


def plot_joint_tracking(
    time_seq_dict,
    time_seq_ref,
    joint_angle_dict,
    joint_angle_ref_dict,
    save_path,
    file_name="joint_angle_tracking",
    file_suffix="",
    title_list=None,
    motor_params=None,
    colors_dict=None,
):
    all_angles = np.concatenate(
        list(joint_angle_dict.values()) + list(joint_angle_ref_dict.values())
    )
    global_ymin, global_ymax = np.min(all_angles) - 0.1, np.max(all_angles) + 0.1

    x_list = []
    y_list = []
    legend_labels = []
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

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axs.flat):
        if i >= n_plots:
            ax.set_visible(False)
            continue

        ax.set_ylim(global_ymin, global_ymax)
        if motor_params is not None and colors_dict is not None:
            ax.set_facecolor(colors_dict[motor_params[legend_labels[2 * i]].brand])

        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{legend_labels[2*i]}" if title_list is None else title_list[i],
            x_label="Time (s)",
            y_label="Position (rad)",
            save_config=True if i == n_plots - 1 else False,
            save_path=save_path if i == n_plots - 1 else None,
            file_name=file_name if i == n_plots - 1 else None,
            file_suffix=file_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
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

        if isinstance(y[0], list):  # Multiple lines
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
