import os

os.environ["USE_JAX"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import argparse
import functools
import importlib
import io
import pickle
import pkgutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
import lz4.frame
import mediapy as media
import mujoco
import numpy as np
import numpy.typing as npt
import torch
from brax import base
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv, get_env_class
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import parse_value

jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGH)


class DomainRandomizationVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[System], Tuple[System, System]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes, self.sys_dict, self.in_axes_dict = randomization_fn(
            self.sys
        )

    def _env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(self._sys_v, state, action)
        return res

    def get_dr_params(self):
        return self.sys_dict, self.in_axes_dict


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


# def convert_state_to_torch(state: State):
#     """
#     Convert a State instance (with JAX arrays) to a dictionary
#     with PyTorch tensors.
#     """
#     state_dict = {}

#     # Convert jax.Array to torch.Tensor by first converting to numpy
#     state_dict["obs"] = torch.from_numpy(np.array(state.obs))
#     state_dict["reward"] = torch.from_numpy(np.array(state.reward))
#     state_dict["done"] = torch.from_numpy(np.array(state.done))

#     # For optional fields, check for None
#     if state.privileged_obs is not None:
#         state_dict["privileged_obs"] = torch.from_numpy(np.array(state.privileged_obs))
#     else:
#         state_dict["privileged_obs"] = None

#     state_dict["metrics"] = {
#         key: torch.from_numpy(np.array(val)) for key, val in state.metrics.items()
#     }

#     # For info, which may be non-numeric, store as is
#     state_dict["info"] = {}
#     for key, val in state.info.items():
#         if not isinstance(val, jax.Array):
#             continue

#         if val.dtype == jnp.uint32:
#             val = val.astype(jnp.int32)

#         state_dict["info"][key] = torch.from_numpy(np.array(val))

#     # Convert basic JAX arrays for joint positions and velocities.
#     state_dict["q"] = torch.from_numpy(np.array(state.pipeline_state.q))
#     state_dict["qd"] = torch.from_numpy(np.array(state.pipeline_state.qd))

#     # Convert the Transform (x) which includes position and rotation.
#     state_dict["x"] = {
#         "pos": torch.from_numpy(np.array(state.pipeline_state.x.pos)),
#         "rot": torch.from_numpy(np.array(state.pipeline_state.x.rot)),
#     }

#     # Convert the Motion (xd) which includes angular and linear velocity.
#     state_dict["xd"] = {
#         "ang": torch.from_numpy(np.array(state.pipeline_state.xd.ang)),
#         "vel": torch.from_numpy(np.array(state.pipeline_state.xd.vel)),
#     }

#     return state_dict


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
        if body_mass_attr_range is None:
            body_mass_attr = {
                "body_mass": sys.body_mass,
                "body_inertia": sys.body_inertia,
                "body_invweight0": sys.body_invweight0,
                "body_subtreemass": sys.body_subtreemass,
                "dof_M0": sys.dof_M0,
                "dof_invweight0": sys.dof_invweight0,
                "tendon_invweight0": sys.tendon_invweight0,
            }
        else:
            body_mass_attr = {
                "body_mass": body_mass_attr_range["body_mass"][0],
                "body_inertia": body_mass_attr_range["body_inertia"][0],
                "body_invweight0": body_mass_attr_range["body_invweight0"][0],
                "body_subtreemass": body_mass_attr_range["body_subtreemass"][0],
                "dof_M0": body_mass_attr_range["dof_M0"][0],
                "dof_invweight0": body_mass_attr_range["dof_invweight0"][0],
                "tendon_invweight0": body_mass_attr_range["tendon_invweight0"][0],
            }
            body_mass_attr_range["body_mass"] = body_mass_attr_range["body_mass"][1:]
            body_mass_attr_range["body_inertia"] = body_mass_attr_range["body_inertia"][
                1:
            ]
            body_mass_attr_range["body_invweight0"] = body_mass_attr_range[
                "body_invweight0"
            ][1:]
            body_mass_attr_range["body_subtreemass"] = body_mass_attr_range[
                "body_subtreemass"
            ][1:]
            body_mass_attr_range["dof_M0"] = body_mass_attr_range["dof_M0"][1:]
            body_mass_attr_range["dof_invweight0"] = body_mass_attr_range[
                "dof_invweight0"
            ][1:]
            body_mass_attr_range["tendon_invweight0"] = body_mass_attr_range[
                "tendon_invweight0"
            ][1:]

        return friction, damping, armature, frictionloss, body_mass_attr, gravity

    friction, damping, armature, frictionloss, body_mass_attr, gravity = rand(rng)
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
        sys = sys.replace(actuator_acc0=body_mass_attr_range["actuator_acc0"][0])
        body_mass_attr_range["actuator_acc0"] = body_mass_attr_range["actuator_acc0"][
            1:
        ]

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(in_axes_dict)
    sys = sys.tree_replace(sys_dict)
    sys = sys.replace(opt=new_opt)
    in_axes = in_axes.replace(opt=in_axes.opt.replace(gravity=0))
    return sys, in_axes, sys_dict, in_axes_dict


def evaluate(
    env: MJXEnv,
    make_networks_factory: Any,
    num_envs: int,
    num_steps: int,
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
    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{env.robot.name}_{env.name}_rollout_{time_str}"
    rollout_path = os.path.join("results", run_name)
    os.makedirs(rollout_path, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, num_envs)

    body_mass_attr_range = None
    if not env.fixed_base:
        body_mass_attr_range = get_body_mass_attr_range(
            env.robot,
            env.cfg.domain_rand.body_mass_range,
            env.cfg.domain_rand.ee_mass_range,
            env.cfg.domain_rand.other_mass_range,
            env.cfg.hang.init_hang_force,
            num_envs,
        )

    domain_randomize_fn = functools.partial(
        domain_randomize,
        friction_range=env.cfg.domain_rand.friction_range,
        damping_range=env.cfg.domain_rand.damping_range,
        armature_range=env.cfg.domain_rand.armature_range,
        frictionloss_range=env.cfg.domain_rand.frictionloss_range,
        gravity_range=env.cfg.domain_rand.gravity_range,
        body_mass_attr_range=body_mass_attr_range,
    )
    v_domain_randomize_fn = functools.partial(domain_randomize_fn, rng=rngs)

    batched_env = DomainRandomizationVmapWrapper(env, v_domain_randomize_fn)

    dr_params = batched_env.get_dr_params()
    # print(dr_params)
    with open(os.path.join(rollout_path, "dr_params.pkl"), "wb") as f:
        pickle.dump(dr_params, f)

    ppo_network = make_networks_factory(
        env.obs_size, env.privileged_obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    policy_path = os.path.join(
        "toddlerbot", "policies", "checkpoints", f"{env.name}_policy"
    )
    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(batched_env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(batched_env.step)
    # jit_step = env.step
    jit_inference_fn = jax.vmap(jax.jit(inference_fn))

    transition_data = {"obs": [], "action": []}

    states = jit_reset(rngs)

    obs_size = env.cfg.obs.num_single_obs
    transition_data["obs"].append(torch.from_numpy(np.array(states.obs[:, :obs_size])))

    times = [time.time()]
    # rollout: List[Any] = [states[0].pipeline_state]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        ctrls, _ = jit_inference_fn(states.obs, rngs)
        states = jit_step(states, ctrls)

        transition_data["obs"].append(
            torch.from_numpy(np.array(states.obs[:, :obs_size]))
        )
        transition_data["action"].append(torch.from_numpy(np.array(ctrls)))

        times.append(time.time())
        # rollout.append(states[0].pipeline_state)
        if i % log_every == 0:
            # Log metrics for the batch – you may average over the batch.
            avg_metrics = jax.tree_util.tree_map(lambda m: jnp.mean(m), states.metrics)
            log_metrics(avg_metrics, times[-1] - times[0])

    transition_data["obs"] = torch.stack(transition_data["obs"])
    transition_data["action"] = torch.stack(transition_data["action"])

    buffer = io.BytesIO()
    torch.save(transition_data, buffer)
    serialized_data = buffer.getvalue()

    # Compress using LZ4
    compressed_data = lz4.frame.compress(serialized_data)

    # Write the compressed data to a file
    with open(os.path.join(rollout_path, "transition_data.pt.lz4"), "wb") as f:
        f.write(compressed_data)

    # try:
    #     render_video(env, rollout, run_name)
    # except Exception:
    #     print("Failed to render the video. Skipped.")


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
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000)
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

    env_cfg.hang.init_hang_force = 0.0
    test_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_noise=True,
        add_domain_rand=True,
        **kwargs,
    )
    # print(
    #     f"training with hang force: {env.cfg.hang.init_hang_force}, eval with hang force: {eval_env.cfg.hang.init_hang_force}"
    # )
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )

    evaluate(test_env, make_networks_factory, args.num_envs, args.num_steps)


if __name__ == "__main__":
    main()
