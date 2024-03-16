import argparse

import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.utils.vis_utils import *

LINE_STYLES = ["-", "--", "-.", ":"]
LINE_COLORS = ["b", "g", "r", "c", "m", "y", "k"]
LINE_MARKERS = ["o", "s", "D", "v", "^", "<", ">"]


def plot_joint_tracking(
    time_seq_dict,
    time_seq_ref,
    joint_angle_dict,
    joint_angle_ref_dict,
    joint2type,
    # joint_range=(-np.pi / 2, np.pi / 2),
):
    x_list = []
    y_list = []
    legend_labels = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        x_list.append(time_seq_ref)
        y_list.append(joint_angle_dict[name])
        y_list.append(joint_angle_ref_dict[name])
        legend_labels.append(name)
        legend_labels.append(name + "_ref")

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    time_suffix = "tracking"
    colors_dict = {
        "dynamixel": "cyan",
        "sunny_sky": "oldlace",
        "mighty_zap": "whitesmoke",
    }
    for i, ax in enumerate(axs.flat):
        # ax.set_ylim(joint_range)
        ax.set_facecolor(colors_dict[joint2type[legend_labels[2 * i]]])
        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{legend_labels[2*i]}",
            x_label="Time (s)",
            y_label="Position (rad)",
            save_config=True if i == len(axs.flat) - 1 else False,
            save_path="results/plots" if i == len(axs.flat) - 1 else None,
            time_suffix=time_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()

    time_str = time.strftime("%Y%m%d_%H%M%S")
    file_name_before = f"{legend_labels[-2]}_{time_suffix}"
    file_name_after = f"joint_angles_{time_suffix}_{time_str}"
    os.rename(
        os.path.join("results/plots", f"{file_name_before}.png"),
        os.path.join("results/plots", f"{file_name_after}.png"),
    )
    os.rename(
        os.path.join("results/plots", f"{file_name_before}_config.pkl"),
        os.path.join("results/plots", f"{file_name_after}_config.pkl"),
    )

    log(
        f"Renamed the files from {file_name_before} to {file_name_after}",
        header="Visualization",
    )


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
    time_suffix=None,
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
        time_suffix=time_suffix,
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
