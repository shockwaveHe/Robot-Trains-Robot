import argparse

import matplotlib.pyplot as plt

from toddlerbot.utils.vis_utils import *

LINE_STYLES = ["-", "--", "-.", ":"]
LINE_COLORS = ["b", "g", "r", "c", "m", "y", "k"]


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
):
    def plot():
        plt.figure(figsize=fig_size)
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
                plt.plot(
                    xi,
                    sub_y,
                    style,
                    color=color,
                    label=legend_labels[i] if legend_labels else None,
                )
        else:  # Single line
            plt.plot(
                x_local,
                y,
                line_styles_local[0],
                color=line_colors_local[0],
                label=legend_labels[0] if legend_labels else None,
            )

        if legend_labels:
            plt.legend()

    # Create and return a visualization function using the make_vis_function
    vis_function = make_vis_function(
        plot,
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
