import os
from copy import deepcopy

os.environ["USE_JAX"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import argparse
import functools
import importlib
import json
import os
import pickle
import pkgutil
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import numpy as np
import numpy.typing as npt
import optax
import torch
import yaml
from brax import base, envs
from brax.io import model
from brax.io.torch import jax_to_torch, torch_to_jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from moviepy.editor import VideoFileClip, clips_array
from orbax import checkpoint as ocp
from tqdm import tqdm

import wandb
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv, get_env_class
from toddlerbot.locomotion.on_policy_runner import OnPolicyRunner
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.locomotion.rsl_wrapper import RSLWrapper
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import dataclass2dict, dump_profiling_data, parse_value

# from toddlerbot.utils.math_utils import soft_clamp

jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGH)


def dynamic_import_envs(env_package: str):
    """Imports all modules from a specified package.

    This function dynamically imports all modules within a given package, allowing their contents to be accessed programmatically. It is useful for loading environment configurations or plugins from a specified package directory.

    Args:
        env_package (str): The name of the package from which to import all modules.
    """
    package = importlib.import_module(env_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{env_package}.{module_name}"
        importlib.import_module(full_module_name)


# Call this to import all policies dynamically
dynamic_import_envs("toddlerbot.locomotion")

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def render_video(
    env: MJXEnv,
    rollout: List[Any],
    run_name: str,
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    """Renders and saves a video of the environment from multiple camera angles.

    Args:
        env (MJXEnv): The environment to render.
        rollout (List[Any]): A list of environment states or actions to render.
        run_name (str): The name of the run, used to organize output files.
        render_every (int, optional): Interval at which frames are rendered from the rollout. Defaults to 2.
        height (int, optional): The height of the rendered video frames. Defaults to 360.
        width (int, optional): The width of the rendered video frames. Defaults to 640.

    Creates:
        A video file for each camera angle ('perspective', 'side', 'top', 'front') and a final concatenated video in a 2x2 grid layout, saved in the 'results' directory under the specified run name.
    """
    # Define paths for each camera's video
    video_paths: List[str] = []

    # Render and save videos for each camera
    for camera in ["perspective", "side", "top", "front"]:
        video_path = os.path.join("results", run_name, f"{camera}.mp4")
        media.write_video(
            video_path,
            env.render(
                rollout[::render_every],
                height=height,
                width=width,
                camera=camera,
                eval=True,
            ),
            fps=1.0 / env.dt / render_every,
        )
        video_paths.append(video_path)

    # Load the video clips using moviepy
    clips = [VideoFileClip(path) for path in video_paths]
    # Arrange the clips in a 2x2 grid
    final_video = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
    # Save the final concatenated video
    final_video.write_videofile(os.path.join("results", run_name, "eval.mp4"))


def log_metrics(
    metrics: Dict[str, Any],
    time_elapsed: float,
    num_steps: int = -1,
    num_total_steps: int = -1,
    width: int = 80,
    pad: int = 35,
):
    """Logs and formats metrics for display, including elapsed time and optional step information.

    Args:
        metrics (Dict[str, Any]): A dictionary containing metric names and their corresponding values.
        time_elapsed (float): The time elapsed since the start of the process.
        num_steps (int, optional): The current number of steps completed. Defaults to -1.
        num_total_steps (int, optional): The total number of steps to be completed. Defaults to -1.
        width (int, optional): The width of the log display. Defaults to 80.
        pad (int, optional): The padding for metric names in the log display. Defaults to 35.

    Returns:
        Dict[str, Any]: A dictionary containing the logged data, including time elapsed and processed metrics.
    """
    log_data: Dict[str, Any] = {"time_elapsed": time_elapsed}
    log_string = f"""{"#" * width}\n"""
    if num_steps >= 0 and num_total_steps > 0:
        log_data["num_steps"] = num_steps
        title = f" \033[1m Learning steps {num_steps}/{num_total_steps} \033[0m "
        log_string += f"""{title.center(width, " ")}\n"""

    for key, value in metrics.items():
        if "std" in key:
            continue

        words = key.split("/")
        if words[0].startswith("eval"):
            if words[1].startswith("episode") and "reward" not in words[1]:
                metric_name = "rew_" + words[1].replace("episode_", "")
            else:
                metric_name = words[1]
        else:
            metric_name = "_".join(words)

        log_data[metric_name] = value
        if (
            "episode_reward" not in metric_name
            and "avg_episode_length" not in metric_name
        ):
            log_string += f"""{f"{metric_name}:":>{pad}} {value:.4f}\n"""

    log_string += (
        f"""{"-" * width}\n""" f"""{"Time elapsed:":>{pad}} {time_elapsed:.1f}\n"""
    )
    if "eval/episode_reward" in metrics:
        log_string += (
            f"""{"Mean reward:":>{pad}} {metrics["eval/episode_reward"]:.3f}\n"""
        )
    if "eval/avg_episode_length" in metrics:
        log_string += f"""{"Mean episode length:":>{pad}} {metrics["eval/avg_episode_length"]:.3f}\n"""

    if "hang_force" in metrics:
        log_string += f"""{"Hang force:":>{pad}} {metrics["hang_force"]:.3f}\n"""
    if "episode_num" in metrics:
        log_string += f"""{"Episode num:":>{pad}} {metrics["episode_num"]}\n"""

    if num_steps > 0 and num_total_steps > 0:
        log_string += (
            f"""{"Computation:":>{pad}} {(num_steps / time_elapsed):.1f} steps/s\n"""
            f"""{"ETA:":>{pad}} {(time_elapsed / num_steps) * (num_total_steps - num_steps):.1f}s\n"""
        )

    print(log_string)

    return log_data


def get_body_mass_attr_range(
    robot: Robot,
    body_mass_range: List[float],
    ee_mass_range: List[float],
    other_mass_range: List[float],
    init_hang_force: float,
    num_envs: int,
):
    """Generates a range of body mass attributes for a robot across multiple environments.

    This function modifies the body mass and inertia of a robot model based on specified
    ranges for different body parts (torso, end-effector, and others) and returns a dictionary
    containing the updated attributes for each environment.

    Args:
        robot (Robot): The robot object containing configuration and name.
        body_mass_range (List[float]): The range of mass deltas for the torso.
        ee_mass_range (List[float]): The range of mass deltas for the end-effector.
        other_mass_range (List[float]): The range of mass deltas for other body parts.
        num_envs (int): The number of environments to generate.

    Returns:
        Dict[str, jax.Array | npt.NDArray[np.float32]]: A dictionary with keys representing
        different body mass attributes and values as JAX arrays or NumPy arrays containing
        the attribute values across all environments.
    """

    suffix = "_hang_scene.xml" if init_hang_force > 0 else "_scene.xml"
    xml_path: str = find_robot_file_path(robot.name, suffix=suffix)
    torso_name = "torso"
    ee_name = robot.config["general"]["ee_name"]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    body_mass = model.body_mass.copy()
    body_inertia = model.body_inertia.copy()

    body_mass_delta_list = np.linspace(body_mass_range[0], body_mass_range[1], num_envs)
    ee_mass_delta_list = np.linspace(ee_mass_range[0], ee_mass_range[1], num_envs)
    other_mass_delta_list = np.linspace(
        other_mass_range[0], other_mass_range[1], num_envs
    )
    # Randomize the order of the body mass deltas
    body_mass_delta_list = np.random.permutation(body_mass_delta_list)
    ee_mass_delta_list = np.random.permutation(ee_mass_delta_list)
    other_mass_delta_list = np.random.permutation(other_mass_delta_list)

    # Create lists to store attributes for all environments
    body_mass_list = []
    body_inertia_list = []
    actuator_acc0_list = []
    body_invweight0_list = []
    body_subtreemass_list = []
    dof_M0_list = []
    dof_invweight0_list = []
    tendon_invweight0_list = []
    for body_mass_delta, ee_mass_delta, other_mass_delta in zip(
        body_mass_delta_list, ee_mass_delta_list, other_mass_delta_list
    ):
        # Update body mass and inertia in the model
        for i in range(model.nbody):
            body_name = model.body(i).name

            if body_mass[i] < 1e-6 or body_mass[i] < other_mass_range[1]:
                continue

            if torso_name in body_name:
                mass_delta = body_mass_delta
            elif ee_name in body_name:
                mass_delta = ee_mass_delta
            else:
                mass_delta = other_mass_delta

            model.body(body_name).mass = body_mass[i] + mass_delta
            model.body(body_name).inertia = (
                (body_mass[i] + mass_delta) / body_mass[i] * body_inertia[i]
            )

        mujoco.mj_setConst(model, data)

        # Append the values to corresponding lists
        body_mass_list.append(jnp.array(model.body_mass))
        body_inertia_list.append(jnp.array(model.body_inertia))
        actuator_acc0_list.append(np.array(model.actuator_acc0))
        body_invweight0_list.append(jnp.array(model.body_invweight0))
        body_subtreemass_list.append(jnp.array(model.body_subtreemass))
        dof_M0_list.append(jnp.array(model.dof_M0))
        dof_invweight0_list.append(jnp.array(model.dof_invweight0))
        tendon_invweight0_list.append(jnp.array(model.tendon_invweight0))

    # Return a dictionary where each key has a JAX array of all values across environments
    body_mass_attr_range: Dict[str, jax.Array | npt.NDArray[np.float32]] = {
        "body_mass": jnp.stack(body_mass_list),
        "body_inertia": jnp.stack(body_inertia_list),
        "actuator_acc0": np.stack(actuator_acc0_list),
        "body_invweight0": jnp.stack(body_invweight0_list),
        "body_subtreemass": jnp.stack(body_subtreemass_list),
        "dof_M0": jnp.stack(dof_M0_list),
        "dof_invweight0": jnp.stack(dof_invweight0_list),
        "tendon_invweight0": jnp.stack(tendon_invweight0_list),
    }

    return body_mass_attr_range


def domain_randomize(
    sys: base.System,
    rng: jax.Array,
    exp_folder_path: str,
    friction_range: List[float],
    damping_range: List[float],
    armature_range: List[float],
    frictionloss_range: List[float],
    gravity_range: List[float],
    body_mass_attr_range: Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]],
) -> Tuple[base.System, base.System]:
    """Randomizes the physical parameters of a system within specified ranges.

    Args:
        sys (base.System): The system whose parameters are to be randomized.
        rng (jax.Array): Random number generator state.
        friction_range (List[float]): Range for randomizing friction values.
        damping_range (List[float]): Range for randomizing damping values.
        armature_range (List[float]): Range for randomizing armature values.
        frictionloss_range (List[float]): Range for randomizing friction loss values.
        body_mass_attr_range (Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]]): Optional dictionary specifying ranges for body mass attributes.

    Returns:
        Tuple[base.System, base.System]: A tuple containing the randomized system and the in_axes configuration for JAX transformations.
    """

    @jax.vmap
    def rand(rng: jax.Array):
        _, rng_friction, rng_damping, rng_armature, rng_frictionloss, rng_gravity = (
            jax.random.split(rng, 6)
        )

        friction = jax.random.uniform(
            rng_friction, (1,), minval=friction_range[0], maxval=friction_range[1]
        )
        friction = sys.geom_friction.at[:, 0].set(friction)

        damping = (
            jax.random.uniform(
                rng_damping, (sys.nv,), minval=damping_range[0], maxval=damping_range[1]
            )
            * sys.dof_damping
        )

        armature = (
            jax.random.uniform(
                rng_armature,
                (sys.nv,),
                minval=armature_range[0],
                maxval=armature_range[1],
            )
            * sys.dof_armature
        )

        frictionloss = (
            jax.random.uniform(
                rng_frictionloss,
                (sys.nv,),
                minval=frictionloss_range[0],
                maxval=frictionloss_range[1],
            )
            * sys.dof_frictionloss
        )
        gravity = (
            jax.random.uniform(
                rng_gravity,
                shape=(),
                minval=gravity_range[0],
                maxval=gravity_range[1],
            )
            * sys.opt.gravity
        )
        return friction, damping, armature, frictionloss, gravity

    friction, damping, armature, frictionloss, gravity = rand(rng)

    body_mass_attr = {}
    if body_mass_attr_range is not None:
        for k, v in body_mass_attr_range.items():
            if isinstance(v, jnp.ndarray):
                body_mass_attr[k] = v[: rng.shape[0]]

    new_opt = sys.opt.replace(gravity=gravity)

    in_axes_dict = {
        "geom_friction": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "dof_frictionloss": 0,
        **{key: 0 for key in body_mass_attr.keys()},
    }

    sys_dict = {
        "geom_friction": friction,
        "dof_damping": damping,
        "dof_armature": armature,
        "dof_frictionloss": frictionloss,
        **body_mass_attr,
    }

    if body_mass_attr_range is not None:
        sys = sys.replace(
            actuator_acc0=body_mass_attr_range["actuator_acc0"][: rng.shape[0]]
        )

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(in_axes_dict)
    sys = sys.tree_replace(sys_dict)
    sys = sys.replace(opt=new_opt)
    in_axes = in_axes.replace(opt=in_axes.opt.replace(gravity=0))

    with open(os.path.join(exp_folder_path, "dr_params.pkl"), "wb") as f:
        pickle.dump((sys_dict, in_axes_dict), f)

    return sys, in_axes


def domain_randomize_from_file(
    sys: base.System,
    rng: jax.Array,
    sys_dict: Dict[str, Any],
    in_axes_dict: Dict[str, Any],
    # embeddings: jax.Array,
) -> Tuple[base.System, base.System]:
    for key in sys_dict:
        sys_dict[key] = sys_dict[key][: rng.shape[0]]

    # embeddings = embeddings[: rng.shape[0]]
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(in_axes_dict)
    sys = sys.tree_replace(sys_dict)
    return sys, in_axes


def load_runner_config(train_cfg: PPOConfig):
    with open("toddlerbot/locomotion/rsl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        for key, value in config["runner"].items():
            config[key] = value

        del config["runner"]

        config["policy"]["actor_hidden_dims"] = train_cfg.policy_hidden_layer_sizes
        config["policy"]["critic_hidden_dims"] = train_cfg.value_hidden_layer_sizes
        config["max_iterations"] = train_cfg.num_timesteps // (
            train_cfg.num_envs * train_cfg.unroll_length
        )
        config["num_steps_per_env"] = train_cfg.unroll_length
        config["algorithm"]["gamma"] = train_cfg.discounting
        config["algorithm"]["num_learning_epochs"] = train_cfg.num_updates_per_batch
        config["algorithm"]["learning_rate"] = train_cfg.learning_rate
        config["algorithm"]["entropy_coef"] = train_cfg.entropy_cost
        config["algorithm"]["clip_param"] = train_cfg.clipping_epsilon
        config["algorithm"]["num_mini_batches"] = train_cfg.num_minibatches
        config["seed"] = train_cfg.seed

    return config


def train(
    env: MJXEnv,
    eval_env: MJXEnv,
    make_networks_factory: Any,
    train_cfg: PPOConfig,
    run_name: str,
    restore_path: str,
    use_symmetry: bool,
    dynamics_time_str: str,
    use_torch: bool,
):
    """Trains a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm.

    This function sets up the training environment, initializes configurations, and manages the training process, including saving configurations, logging metrics, and handling checkpoints.

    Args:
        env (MJXEnv): The training environment.
        eval_env (MJXEnv): The evaluation environment.
        make_networks_factory (Any): Factory function to create neural network models.
        train_cfg (PPOConfig): Configuration settings for the PPO training process.
        run_name (str): Name of the training run, used for organizing results.
        restore_path (str): Path to restore a previous checkpoint, if any.
    """
    exp_folder_path = os.path.join("results", run_name)
    os.makedirs(exp_folder_path, exist_ok=True)

    restore_checkpoint_path = (
        os.path.abspath(restore_path) if len(restore_path) > 0 else None
    )

    # Save train config to a file and print it
    train_config_dict = dataclass2dict(train_cfg)  # Convert dataclass to dictionary
    with open(os.path.join(exp_folder_path, "train_config.json"), "w") as f:
        json.dump(train_config_dict, f, indent=4)

    # Print the train config
    print("Train Config:")
    print(json.dumps(train_config_dict, indent=4))  # Pretty-print the config

    # Save env config to a file and print it
    env_config_dict = dataclass2dict(env.cfg)  # Convert dataclass to dictionary
    with open(os.path.join(exp_folder_path, "env_config.json"), "w") as f:
        json.dump(env_config_dict, f, indent=4)

    # Print the env config
    print("Env Config:")
    print(json.dumps(env_config_dict, indent=4))  # Pretty-print the config

    # Copy the Python scripts
    shutil.copytree(
        os.path.join("toddlerbot", "locomotion"),
        os.path.join(exp_folder_path, "locomotion"),
    )

    domain_randomize_fn = None
    if env.add_domain_rand:
        if len(dynamics_time_str) > 0:
            dynamics_dir = os.path.join(
                "results", f"dynamics_model_{dynamics_time_str}"
            )
            with open(os.path.join(dynamics_dir, "dr_params.pkl"), "rb") as f:
                dr_params = pickle.load(f)

            sys_dict = {
                key: value[: train_cfg.num_envs] for key, value in dr_params[0].items()
            }
            in_axes_dict = dr_params[1]

            # if env.cfg.obs.latent_dim == 2048:
            #     embeddings = np.load(os.path.join(latent_dir, "embeddings.npy"))
            # else:
            #     flattened_params = [
            #         v.reshape(v.shape[0], -1) for v in dr_params[0].values()
            #     ]
            #     embeddings = np.concatenate(flattened_params, axis=1)

            # embeddings = (
            #     embeddings[: train_cfg.num_envs]
            #     / np.linalg.norm(embeddings, axis=-1).mean()
            # )
            domain_randomize_fn = functools.partial(
                domain_randomize_from_file,
                sys_dict=sys_dict,
                in_axes_dict=in_axes_dict,
                # embeddings=embeddings,
            )

        else:
            body_mass_attr_range = None
            if not env.fixed_base:
                body_mass_attr_range = get_body_mass_attr_range(
                    env.robot,
                    env.cfg.domain_rand.body_mass_range,
                    env.cfg.domain_rand.ee_mass_range,
                    env.cfg.domain_rand.other_mass_range,
                    env.cfg.hang.init_hang_force,
                    train_cfg.num_envs,
                )

            domain_randomize_fn = functools.partial(
                domain_randomize,
                exp_folder_path=exp_folder_path,
                friction_range=env.cfg.domain_rand.friction_range,
                damping_range=env.cfg.domain_rand.damping_range,
                armature_range=env.cfg.domain_rand.armature_range,
                frictionloss_range=env.cfg.domain_rand.frictionloss_range,
                gravity_range=env.cfg.domain_rand.gravity_range,
                body_mass_attr_range=body_mass_attr_range,
            )

    if use_torch:
        # The number of environment steps executed for every training step.
        key = jax.random.PRNGKey(train_cfg.seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, jax.process_index())
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        # key_networks should be global, so that networks are initialized the same
        # way for different processes.
        # key_policy, key_value = jax.random.split(global_key)
        del global_key

        v_randomization_fn = None
        if domain_randomize_fn is not None:
            randomization_rng = jax.random.split(key_env, train_cfg.num_envs)
            v_randomization_fn = functools.partial(
                domain_randomize_fn, rng=randomization_rng
            )

        # if isinstance(environment, envs.Env):
        wrap_for_training = envs.training.wrap
        # else:
        #     wrap_for_training = envs_v1.wrappers.wrap_for_training

        env = wrap_for_training(
            env,
            episode_length=train_cfg.episode_length,
            randomization_fn=v_randomization_fn,
        )

        # TODO: Save data when the base policy is working
        rsl_env = RSLWrapper(
            env, device="cuda:0", exp_folder_path="", train_cfg=train_cfg
        )
        runner_config = load_runner_config(train_cfg)
        if len(dynamics_time_str) > 0:
            with open(
                os.path.join("toddlerbot", "autoencoder", "config.yaml"), "r"
            ) as f:
                autoencoder_config = yaml.safe_load(f)
                autoencoder_config["data"]["time_str"] = dynamics_time_str
                runner_config["algorithm"]["autoencoder_cfg"] = autoencoder_config

        if not use_symmetry and "symmetry_cfg" in runner_config["algorithm"]:
            del runner_config["algorithm"]["symmetry_cfg"]

        # Print the env config
        print("Runner Config:")
        print(json.dumps(runner_config, indent=4))  # Pretty-print the config

        runner = OnPolicyRunner(rsl_env, runner_config, run_name, device="cuda:0")

        if len(dynamics_time_str) > 0:
            dynamics_path = os.path.join(
                "results", f"dynamics_model_{dynamics_time_str}"
            )
            with open(os.path.join(dynamics_path, "summary.json"), "r") as f:
                summary = json.load(f)

            base_policy_path = os.path.join(
                "results", summary["run_name"], "model_best.pt"
            )
            loaded_dict = torch.load(base_policy_path, weights_only=False)

            old_model_dict = loaded_dict["model_state_dict"]
            model_dict = runner.alg.actor_critic.state_dict()
            new_dict = {}
            for name, param in model_dict.items():
                if name in old_model_dict:
                    loaded_param = old_model_dict[name]

                    if param.shape != loaded_param.shape:
                        print(
                            f"Shape mismatch for '{name}': model {param.shape}, loaded {loaded_param.shape}"
                        )
                        # Assuming first dimension mismatch (common for linear layers' weight matrices)
                        if param.ndim == 2 and param.shape[1] > loaded_param.shape[1]:
                            # Calculate padding size
                            padding_size = param.shape[1] - loaded_param.shape[1]

                            # Pad loaded param with zeros
                            padded_param = torch.nn.functional.pad(
                                loaded_param,
                                (0, padding_size, 0, 0),  # Pad last dimension on right
                                mode="constant",
                                value=0,
                            )
                            new_dict[name] = padded_param
                            print(
                                f"Padded '{name}' with zeros to match shape {padded_param.shape}"
                            )
                        else:
                            # Handle other mismatches if necessary
                            new_dict[name] = param  # Use original param
                            print(
                                f"Using original param for '{name}' due to unsupported shape mismatch."
                            )
                    else:
                        new_dict[name] = loaded_param
                else:
                    # If param not in loaded dict, use original param
                    new_dict[name] = param
                    print(
                        f"'{name}' not found in loaded state dict. Using original param."
                    )

            runner.alg.actor_critic.load_state_dict(new_dict)
            # runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # runner.current_learning_iteration = loaded_dict["iter"]
            # print(loaded_dict["infos"])

        try:
            runner.learn(
                num_learning_iterations=runner_config["max_iterations"],
                init_at_random_ep_len=False,
            )
        except KeyboardInterrupt:
            prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
            dump_profiling_data(prof_path)

    else:
        wandb.init(
            project="ToddlerBot",
            sync_tensorboard=True,
            name=run_name,
            config=dataclass2dict(train_cfg),
        )

        orbax_checkpointer = ocp.PyTreeCheckpointer()

        def policy_params_fn(current_step: int, make_policy: Any, params: Any):
            # save checkpoints
            save_args = orbax_utils.save_args_from_target(params)
            path = os.path.abspath(os.path.join(exp_folder_path, f"{current_step}"))
            orbax_checkpointer.save(path, params, force=True, save_args=save_args)
            policy_path = os.path.join(path, "policy")
            model.save_params(policy_path, (params[0], params[1].policy))

        learning_rate_schedule_fn = optax.cosine_decay_schedule(
            train_cfg.learning_rate,
            train_cfg.decay_steps,
            train_cfg.alpha,
        )

        train_fn = functools.partial(
            ppo.train,
            num_timesteps=train_cfg.num_timesteps,
            num_evals=train_cfg.num_evals,
            episode_length=train_cfg.episode_length,
            unroll_length=train_cfg.unroll_length,
            num_minibatches=train_cfg.num_minibatches,
            num_updates_per_batch=train_cfg.num_updates_per_batch,
            discounting=train_cfg.discounting,
            learning_rate=train_cfg.learning_rate,
            learning_rate_schedule_fn=learning_rate_schedule_fn,
            entropy_cost=train_cfg.entropy_cost,
            clipping_epsilon=train_cfg.clipping_epsilon,
            num_envs=train_cfg.num_envs,
            batch_size=train_cfg.batch_size,
            seed=train_cfg.seed,
            network_factory=make_networks_factory,
            randomization_fn=domain_randomize_fn,
            render_interval=train_cfg.render_interval,
            policy_params_fn=policy_params_fn,
            restore_checkpoint_path=restore_checkpoint_path,
            run_name=run_name,
        )

        times = [time.time()]

        last_ckpt_step = 0
        best_ckpt_step = 0
        best_episode_reward = -float("inf")

        def progress(num_steps: int, metrics: Dict[str, Any]):
            nonlocal best_episode_reward, best_ckpt_step, last_ckpt_step

            times.append(time.time())

            if last_ckpt_step > 0:
                shutil.copy2(
                    os.path.join(exp_folder_path, str(last_ckpt_step), "policy"),
                    os.path.join(exp_folder_path, "policy"),
                )

            last_ckpt_step = num_steps

            episode_reward = float(metrics.get("eval/episode_reward", 0.0))
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_ckpt_step = num_steps

            log_data = log_metrics(
                metrics, times[-1] - times[0], num_steps, train_cfg.num_timesteps
            )

            # Log metrics to wandb
            wandb.log(log_data)

        try:
            _, params, _ = train_fn(
                environment=env, eval_env=eval_env, progress_fn=progress
            )
        except KeyboardInterrupt:
            prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
            dump_profiling_data(prof_path)

        shutil.copy2(
            os.path.join(exp_folder_path, str(best_ckpt_step), "policy"),
            os.path.join(exp_folder_path, "best_policy"),
        )

        print(f"time to jit: {times[1] - times[0]}")
        print(f"time to train: {times[-1] - times[1]}")
        print(f"best checkpoint step: {best_ckpt_step}")
        print(f"best episode reward: {best_episode_reward}")


def evaluate(
    env: MJXEnv,
    make_networks_factory: Any,
    train_cfg: PPOConfig,
    run_name: str,
    use_symmetry: bool,
    dynamics_time_str: str,
    use_torch: bool,
    num_steps: int = 1000,
    log_every: int = 100,
):
    """Evaluates a policy in a given environment using a specified network factory and logs the results.

    Args:
        env (MJXEnv): The environment in which the policy is evaluated.
        make_networks_factory (Any): A factory function to create network architectures for the policy.
        run_name (str): The name of the run, used for saving and loading policy parameters.
        num_steps (int, optional): The number of steps to evaluate the policy. Defaults to 1000.
        log_every (int, optional): The frequency (in steps) at which metrics are logged. Defaults to 100.
    """
    rng = jax.random.PRNGKey(0)

    # if len(env.cfg.obs.latent_folder) > 0:
    #     latent_dir = os.path.join("results", env.cfg.obs.latent_folder)

    #     if env.cfg.obs.latent_dim == 2048:
    #         embeddings = np.load(os.path.join(latent_dir, "embeddings.npy"))
    #     else:
    #         with open(os.path.join(latent_dir, "dr_params.pkl"), "rb") as f:
    #             dr_params = pickle.load(f)

    #         flattened_params = [
    #             v.reshape(v.shape[0], -1) for v in dr_params[0].values()
    #         ]
    #         embeddings = np.concatenate(flattened_params, axis=1)

    #     env.embedding = embeddings[0] / np.linalg.norm(embeddings, axis=-1).mean()

    if use_torch:
        rsl_env = RSLWrapper(env, device="cuda:0")
        runner_config = load_runner_config(train_cfg)
        if len(dynamics_time_str) > 0:
            with open(
                os.path.join("toddlerbot", "autoencoder", "config.yaml"), "r"
            ) as f:
                autoencoder_config = yaml.safe_load(f)
                autoencoder_config["data"]["time_str"] = dynamics_time_str
                runner_config["algorithm"]["autoencoder_cfg"] = autoencoder_config

        if not use_symmetry and "symmetry_cfg" in runner_config["algorithm"]:
            del runner_config["algorithm"]["symmetry_cfg"]

        runner = OnPolicyRunner(
            rsl_env, runner_config, os.path.join("results", run_name), device="cuda:0"
        )
        policy_path = os.path.join("results", run_name, "model_best.pt")
        runner.load(policy_path)
        inference_fn = runner.get_inference_policy(device=rsl_env.device)

        def policy(obs: jax.Array):
            obs_torch = jax_to_torch(obs, device=rsl_env.device)
            action_torch = inference_fn(obs_torch)
            # action_torch = soft_clamp(action_torch, -1.0, 1.0)
            action = torch_to_jax(action_torch)
            return action

    else:
        ppo_network = make_networks_factory(
            env.obs_size, env.privileged_obs_size, env.action_size
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        policy_path = os.path.join("results", run_name, "best_policy")
        if not os.path.exists(policy_path):
            policy_path = os.path.join("results", run_name, "policy")

        params = model.load_params(policy_path)
        inference_fn = jax.jit(make_policy(params, deterministic=True))

        def policy(obs: jax.Array):
            return inference_fn(obs, rng)[0]

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(rng)

    times = [time.time()]
    rollout: List[Any] = [state.pipeline_state]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        ctrl = policy(state.obs)
        state = jit_step(state, ctrl)
        times.append(time.time())
        rollout.append(state.pipeline_state)
        if i % log_every == 0:
            log_metrics(state.metrics, times[-1] - times[0])

    try:
        render_video(env, rollout, run_name)
        wandb.log(
            {
                "video": wandb.Video(
                    os.path.join("results", run_name, "eval.mp4"), format="mp4"
                )
            }
        )
    except Exception:
        print("Failed to render the video. Skipped.")


def main(args=None):
    """Trains or evaluates a policy for a specified robot and environment using PPO.

    This function sets up the training or evaluation of a policy for a robot in a specified environment. It parses command-line arguments to configure the robot, environment, evaluation settings, and other parameters. It then loads configuration files, binds any overridden parameters, and initializes the environment and robot. Depending on the arguments, it either trains a new policy or evaluates an existing one.

    Args:
        args (list, optional): List of command-line arguments. If None, arguments are parsed from sys.argv.

    Raises:
        FileNotFoundError: If a specified gin configuration file or evaluation run is not found.
    """
    parser = argparse.ArgumentParser(description="Train the mjx policy.")
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
    parser.add_argument(
        "--eval",
        type=str,
        default="",
        help="Provide the time string of the run to evaluate.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default="",
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="",
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--gin-files",
        type=str,
        default="",
        help="List of gin config files",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default="",
        help="Override config parameters (e.g., SimConfig.timestep=0.01 ObsConfig.frame_stack=10)",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="",
        help="time str of the dynamics model.",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        default=False,
        help="Use torch instead of jax.",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        default=False,
        help="Use symmetry.",
    )

    args = parser.parse_args()

    gin_file_list = [args.env] + args.gin_files.split(" ")
    for gin_file in gin_file_list:
        if len(gin_file) == 0:
            continue

        gin_file_path = os.path.join(
            os.path.dirname(__file__),
            gin_file + ".gin" if not gin_file.endswith(".gin") else gin_file,
        )
        if not os.path.exists(gin_file_path):
            raise FileNotFoundError(f"File {gin_file_path} not found.")

        gin.parse_config_file(gin_file_path)

    # Bind parameters from --config_override
    if len(args.config_override) > 0:
        for override in args.config_override.split(","):
            key, value = override.split("=", 1)  # Split into key-value pair
            gin.bind_parameter(key, parse_value(value))

    robot = Robot(args.robot)

    EnvClass = get_env_class(args.env)
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    kwargs = {}
    if len(args.ref) > 0:
        kwargs = {"ref_motion_type": args.ref}

    if "fixed" in args.env:
        train_cfg.num_timesteps = 20_000_000
        train_cfg.num_evals = 200

        env_cfg.rewards.healthy_z_range = [-0.2, 0.2]
        env_cfg.rewards.scales.reset()

        if "walk" in args.env:
            env_cfg.rewards.scales.feet_distance = 0.5

        env_cfg.rewards.scales.leg_motor_pos = 5.0
        env_cfg.rewards.scales.waist_motor_pos = 5.0
        env_cfg.rewards.scales.motor_torque = 5e-2
        env_cfg.rewards.scales.leg_action_rate = 1e-2
        env_cfg.rewards.scales.leg_action_acc = 1e-2
        env_cfg.rewards.scales.waist_action_rate = 1e-2
        env_cfg.rewards.scales.waist_action_acc = 1e-2

    env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_noise=env_cfg.noise.add_noise,
        add_domain_rand=env_cfg.domain_rand.add_domain_rand,
        **kwargs,  # type: ignore
    )

    eval_env_cfg = deepcopy(env_cfg)
    eval_env_cfg.hang.init_hang_force = 0.0
    eval_env = EnvClass(
        args.env,
        robot,
        eval_env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_noise=env_cfg.noise.add_noise,
        add_domain_rand=env_cfg.domain_rand.add_domain_rand,
        **kwargs,  # type: ignore
    )
    test_env = EnvClass(
        args.env,
        robot,
        eval_env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_noise=False,
        add_domain_rand=False,
        **kwargs,
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )

    if len(args.eval) > 0:
        time_str = args.eval
    else:
        time_str = time.strftime("%Y%m%d_%H%M%S")

    config_override_str: str = (
        "" if len(args.config_override) == 0 else f"_{args.config_override}"
    )
    run_name = f"{robot.name}_{args.env}_ppo{config_override_str}_{time_str}"
    run_name += "_latent" if len(args.dynamics) > 0 else ""
    run_name += "_torch" if args.torch else ""

    if len(args.eval) > 0:
        if os.path.exists(os.path.join("results", run_name)):
            evaluate(
                test_env,
                make_networks_factory,
                train_cfg,
                run_name,
                args.symmetry,
                args.dynamics,
                args.torch,
            )
        else:
            raise FileNotFoundError(f"Run {args.eval} not found.")
    else:
        train(
            env,
            eval_env,
            make_networks_factory,
            train_cfg,
            run_name,
            args.restore,
            args.symmetry,
            args.dynamics,
            args.torch,
        )
        evaluate(
            test_env,
            make_networks_factory,
            train_cfg,
            run_name,
            args.symmetry,
            args.dynamics,
            args.torch,
        )


if __name__ == "__main__":
    main()
