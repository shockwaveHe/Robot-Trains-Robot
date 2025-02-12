import os
import sys
import shutil
import platform


def get_env_path():
    """Determines the path of the current Python environment.

    Returns:
        str: The path to the current Python environment. If a virtual environment is active, returns the virtual environment's path. If a conda environment is active, returns the conda environment's path. Otherwise, returns the system environment's path.
    """
    if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        return sys.prefix
    elif "CONDA_PREFIX" in os.environ:
        return os.environ["CONDA_PREFIX"]
    else:
        return sys.prefix


def modify_config():
    """Modifies config.py in onshape_to_robot package if running on macOS."""

    if platform.system() != "Darwin":
        return

    env_path = get_env_path()

    # Construct the path to config.py
    config_path = os.path.join(
        env_path,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "onshape_to_robot",
        "config.py",
    )

    if not os.path.exists(config_path):
        print(f"Error: config.py not found at {config_path}")
        return

    try:
        # Read the content of config.py
        with open(config_path, "r") as file:
            lines = file.readlines()

        # Search for the specific line and modify it
        modified = False
        for i, line in enumerate(lines):
            if "if not os.path.exists('/usr/bin/meshlabserver') != 0:" in line:
                lines[i] = (
                    "    import shutil\n    if shutil.which('meshlabserver') is None:\n"
                )
                modified = True
                break

        if modified:
            # Create a backup before modifying
            shutil.copy(config_path, config_path + ".bak")

            # Write the modified content back to config.py
            with open(config_path, "w") as file:
                file.writelines(lines)

            print("Modification complete. A backup has been saved as config.py.bak.")
        else:
            print("No changes were made. Target line not found.")

    except Exception as e:
        print(f"Error while modifying the file: {e}")


if __name__ == "__main__":
    modify_config()
