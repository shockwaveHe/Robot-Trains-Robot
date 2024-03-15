import functools
import os
import pickle
import time

import matplotlib.pyplot as plt
import seaborn as sns

from toddlerbot.utils.misc_utils import log

backend_curr = "Agg"
non_interactive_backends = ["Agg", "SVG", "PDF", "PS"]
plt.switch_backend(backend_curr)
sns.set_theme(style="darkgrid")


def make_vis_function(
    func,
    ax=None,
    title=None,
    x_label=None,
    y_label=None,
    save_config=False,
    save_path=None,
    time_suffix=None,
    blocking=True,
):
    """
    Creates a visualization function with pre-configured settings.

    Parameters:
    - func: The base function to be wrapped with visualization and saving logic.
    - blocking, style, title, etc.: Visualization and configuration parameters.

    Returns:
    - A new function that incorporates the specified visualization and saving logic.
    """

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        if time_suffix is None:
            suffix = time.strftime("%Y%m%d_%H%M%S")
        else:
            suffix = time_suffix

        if len(suffix) > 0:
            suffix = f"_{suffix}"

        # Save configuration if requested
        if save_config and save_path:
            config = {
                "function": f"{func.__module__}.{func.__name__}",
                "parameters": kwargs,
            }

            config_file_name = f"{title.lower().replace(' ', '_')}{suffix}_config.pkl"
            config_path = os.path.join(save_path, config_file_name)
            with open(config_path, "wb") as file:
                pickle.dump(config, file)
                log(f"Configuration saved to: {config_path}", header="Visualization")

        if not blocking:
            return

        # Execute the original function
        result = func(*args, **kwargs)

        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(True)
        plt.tight_layout()

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            file_path = os.path.join(
                save_path, f"{title.lower().replace(' ', '_')}{suffix}.png"
            )
            plt.savefig(file_path)
            log(f"Graph saved as: {file_path}", header="Visualization")
        else:
            if backend_curr not in non_interactive_backends:
                plt.show()

        return result

    return wrapped_function


def load_and_run_visualization(config_path):
    """
    Loads visualization parameters from a file and executes the specified visualization function.

    Parameters:
    - config_path: Path to the configuration file.
    - config_type: Type of the configuration file ('json' or 'pickle').
    """
    # Load the configuration based on its type
    if os.path.exists(config_path) and config_path.endswith(".pkl"):
        with open(config_path, "rb") as file:
            config = pickle.load(file)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Dynamically import and call the specified function
    func_module, func_name = config["function"].rsplit(".", 1)
    module = __import__(func_module, fromlist=[func_name])
    func = getattr(module, func_name)
    func(**config["parameters"], blocking=True)
