import argparse
import importlib
import os
import pkgutil

os.environ["USE_JAX"] = "true"

import gin
import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import get_env_class
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot


def dynamic_import_envs(env_package: str):
    """Dynamically import all modules in the given package."""
    package = importlib.import_module(env_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{env_package}.{module_name}"
        importlib.import_module(full_module_name)


# Call this to import all policies dynamically
dynamic_import_envs("toddlerbot.locomotion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="walk",
        help="The name of the env.",
    )
    args = parser.parse_args()

    gin_file_list = [args.env]
    for gin_file in gin_file_list:
        if len(gin_file) == 0:
            continue

        gin_file_path = os.path.join(
            "toddlerbot/locomotion",
            gin_file + ".gin" if not gin_file.endswith(".gin") else gin_file,
        )
        if not os.path.exists(gin_file_path):
            raise FileNotFoundError(f"File {gin_file_path} not found.")

        gin.parse_config_file(gin_file_path)

    robot = Robot(args.robot)

    EnvClass = get_env_class(args.env)
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    kwargs = {}

    test_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_noise=False,
        add_domain_rand=False,
        **kwargs,
    )

    # Sampling multiple commands
    num_samples = 1000
    commands = []
    rng = jax.random.PRNGKey(0)

    for _ in tqdm(range(num_samples), desc="Sampling commands"):
        rng, sub_rng = jax.random.split(rng)
        command = test_env._sample_command(sub_rng)
        commands.append(np.array(command[5:8]))  # x, y, and z walk commands

    # Convert commands to a numpy array for plotting
    commands = np.array(commands)
    x, y, z = commands[:, 0], commands[:, 1], commands[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", alpha=0.7, s=5)

    ax.set_xlabel("x command")
    ax.set_ylabel("y command")
    ax.set_zlabel("z command")
    ax.set_title("Distribution of Sampled Walk and Turn Commands")

    # Add a color bar for z-axis (turn command)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("z command (turning)")

    plt.show()
