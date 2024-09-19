import os

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["USE_JAX"] = "true"

import argparse
import functools
import json
import shutil
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Type

import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import numpy as np
import numpy.typing as npt
import optax
from brax import base
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from moviepy.editor import VideoFileClip, clips_array
from orbax import checkpoint as ocp
from tqdm import tqdm

import wandb
from toddlerbot.envs.balance_env import BalanceCfg, BalanceEnv
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.envs.rotate_torso_env import RotateTorsoCfg, RotateTorsoEnv
from toddlerbot.envs.squat_env import SquatCfg, SquatEnv
from toddlerbot.envs.walk_env import WalkCfg, WalkEnv
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path


def render_video(
    env: MJXEnv,
    rollout: List[Any],
    run_name: str,
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    # Define paths for each camera's video
    video_paths: List[str] = []

    # Render and save videos for each camera
    for camera in ["perspective", "side", "top", "front"]:
        video_path = os.path.join("results", run_name, f"{camera}.mp4")
        media.write_video(
            video_path,
            env.render(
                rollout[::render_every], height=height, width=width, camera=camera
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
    log_data: Dict[str, Any] = {"time_elapsed": time_elapsed}
    log_string = f"""{'#' * width}\n"""
    if num_steps >= 0 and num_total_steps > 0:
        log_data["num_steps"] = num_steps
        title = f" \033[1m Learning steps {num_steps}/{num_total_steps } \033[0m "
        log_string += f"""{title.center(width, ' ')}\n"""

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
            log_string += f"""{f'{metric_name}:':>{pad}} {value:.4f}\n"""

    log_string += (
        f"""{'-' * width}\n""" f"""{'Time elapsed:':>{pad}} {time_elapsed:.1f}\n"""
    )
    if "eval/episode_reward" in metrics:
        log_string += (
            f"""{'Mean reward:':>{pad}} {metrics['eval/episode_reward']:.3f}\n"""
        )
    if "eval/avg_episode_length" in metrics:
        log_string += f"""{'Mean episode length:':>{pad}} {metrics['eval/avg_episode_length']:.3f}\n"""

    if num_steps > 0 and num_total_steps > 0:
        log_string += (
            f"""{'Computation:':>{pad}} {(num_steps / time_elapsed ):.1f} steps/s\n"""
            f"""{'ETA:':>{pad}} {(time_elapsed / num_steps) * (num_total_steps - num_steps):.1f}s\n"""
        )

    print(log_string)

    return log_data


def get_body_mass_attr_range(robot: Robot, body_mass_range: List[float], num_envs: int):
    xml_path: str = find_robot_file_path(robot.name, suffix="_scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    body_mass = np.array(model.body("torso").mass).copy()
    body_inertia = np.array(model.body("torso").inertia).copy()
    body_mass_delta_range = np.linspace(
        body_mass_range[0], body_mass_range[1], num_envs
    )
    # Randomize the order of the body mass deltas
    body_mass_delta_range = np.random.permutation(body_mass_delta_range)

    # Create lists to store attributes for all environments
    body_mass_list = []
    body_inertia_list = []
    actuator_acc0_list = []
    body_invweight0_list = []
    body_subtreemass_list = []
    dof_M0_list = []
    dof_invweight0_list = []
    tendon_invweight0_list = []
    for body_mass_delta in body_mass_delta_range:
        # Update body mass and inertia in the model
        model.body("torso").mass = body_mass + body_mass_delta
        model.body("torso").inertia = (
            (body_mass + body_mass_delta) / body_mass * body_inertia
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
    body_mass_attr_range: Dict[str, jax.Array | npt.NDArray[np.float32]],
) -> Tuple[base.System, base.System]:
    @jax.vmap
    def rand(rng: jax.Array):
        _, key = jax.random.split(rng, 2)

        # Friction
        friction = jax.random.uniform(
            key,
            (1,),
            minval=friction_range[0],
            maxval=friction_range[1],
        )
        friction = sys.geom_friction.at[:, 0].set(friction)

        damping = (
            jax.random.uniform(
                key, (sys.nv,), minval=damping_range[0], maxval=damping_range[1]
            )
            * sys.dof_damping
        )

        armature = (
            jax.random.uniform(
                key, (sys.nv,), minval=armature_range[0], maxval=armature_range[1]
            )
            * sys.dof_armature
        )

        frictionloss = (
            jax.random.uniform(
                key,
                (sys.nv,),
                minval=frictionloss_range[0],
                maxval=frictionloss_range[1],
            )
            * sys.dof_frictionloss
        )

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
        body_mass_attr_range["body_inertia"] = body_mass_attr_range["body_inertia"][1:]
        body_mass_attr_range["body_invweight0"] = body_mass_attr_range[
            "body_invweight0"
        ][1:]
        body_mass_attr_range["body_subtreemass"] = body_mass_attr_range[
            "body_subtreemass"
        ][1:]
        body_mass_attr_range["dof_M0"] = body_mass_attr_range["dof_M0"][1:]
        body_mass_attr_range["dof_invweight0"] = body_mass_attr_range["dof_invweight0"][
            1:
        ]
        body_mass_attr_range["tendon_invweight0"] = body_mass_attr_range[
            "tendon_invweight0"
        ][1:]

        return friction, damping, armature, frictionloss, body_mass_attr

    friction, damping, armature, frictionloss, body_mass_attr = rand(rng)

    in_axes_dict = {
        "geom_friction": 0,
        # "actuator_gainprm": 0,
        # "actuator_biasprm": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "dof_frictionloss": 0,
        **{key: 0 for key in body_mass_attr.keys()},
    }

    sys_dict = {
        "geom_friction": friction,
        # "actuator_gainprm": gain,
        # "actuator_biasprm": bias,
        "dof_damping": damping,
        "dof_armature": armature,
        "dof_frictionloss": frictionloss,
        **body_mass_attr,
    }

    # jax.debug.breakpoint()
    # for key, value in body_mass_attr.items():
    #     in_axes_dict[key] = 0
    #     sys_dict[key] = value

    if body_mass_attr_range is not None:
        sys = sys.replace(actuator_acc0=body_mass_attr_range["actuator_acc0"][0])
        body_mass_attr_range["actuator_acc0"] = body_mass_attr_range["actuator_acc0"][
            1:
        ]

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(in_axes_dict)
    sys = sys.tree_replace(sys_dict)

    return sys, in_axes


def train(
    env: MJXEnv,
    eval_env: MJXEnv,
    make_networks_factory: Any,
    train_cfg: PPOConfig,
    run_name: str,
    restore_path: str,
):
    exp_folder_path = os.path.join("results", run_name)
    os.makedirs(exp_folder_path, exist_ok=True)

    restore_checkpoint_path = (
        os.path.abspath(restore_path) if len(restore_path) > 0 else None
    )

    with open(os.path.join(exp_folder_path, "train_config.json"), "w") as f:
        json.dump(asdict(train_cfg), f, indent=4)

    with open(os.path.join(exp_folder_path, "env_config.json"), "w") as f:
        json.dump(asdict(env.cfg), f, indent=4)

    # Copy the Python scripts
    shutil.copytree(
        os.path.join("toddlerbot", "envs"), os.path.join(exp_folder_path, "envs")
    )

    wandb.init(
        project="ToddlerBot",
        sync_tensorboard=True,
        name=run_name,
        config=asdict(train_cfg),
    )

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        # save checkpoints
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.abspath(os.path.join(exp_folder_path, f"{current_step}"))
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        policy_path = os.path.join(path, "policy")
        model.save_params(policy_path, (params[0], params[1].policy))

    # TODO: Implement adaptive learning rate
    learning_rate_schedule_fn = optax.cosine_decay_schedule(
        train_cfg.learning_rate,
        train_cfg.decay_steps,
        train_cfg.alpha,
    )

    body_mass_attr_range = None
    if env.cfg.domain_rand.added_mass_range is not None and not env.fixed_base:
        body_mass_attr_range = get_body_mass_attr_range(
            env.robot, env.cfg.domain_rand.added_mass_range, train_cfg.num_envs
        )

    domain_randomize_fn = functools.partial(
        domain_randomize,
        friction_range=env.cfg.domain_rand.friction_range,
        damping_range=env.cfg.domain_rand.damping_range,
        armature_range=env.cfg.domain_rand.armature_range,
        frictionloss_range=env.cfg.domain_rand.frictionloss_range,
        body_mass_attr_range=body_mass_attr_range,
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
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=restore_checkpoint_path,
    )

    times = [time.time()]

    best_ckpt_step = 0
    best_episode_reward = -float("inf")

    def progress(num_steps: int, metrics: Dict[str, Any]):
        nonlocal best_episode_reward, best_ckpt_step

        times.append(time.time())

        episode_reward = float(metrics.get("eval/episode_reward", 0.0))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_ckpt_step = num_steps

        log_data = log_metrics(
            metrics, times[-1] - times[0], num_steps, train_cfg.num_timesteps
        )

        # Log metrics to wandb
        wandb.log(log_data)

    _, params, _ = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)

    model_path = os.path.join(exp_folder_path, "policy")
    model.save_params(model_path, params)

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
    run_name: str,
    num_steps: int = 1000,
    log_every: int = 100,
):
    ppo_network = make_networks_factory(
        env.obs_size, env.privileged_obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_path = os.path.join("results", run_name, "best_policy")
    if not os.path.exists(policy_path):
        policy_path = os.path.join("results", run_name, "policy")

    params = model.load_params(policy_path)
    inference_fn = make_policy(params)

    # initialize the state
    jit_reset = jax.jit(env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(env.step)
    # jit_step = env.step
    jit_inference_fn = jax.jit(inference_fn)
    # jit_inference_fn = inference_fn

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    rollout: List[Any] = [state.pipeline_state]

    times = [time.time()]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        times.append(time.time())
        rollout.append(state.pipeline_state)
        if i % log_every == 0:
            log_metrics(state.metrics, times[-1] - times[0])

    try:
        render_video(env, rollout, run_name)
    except Exception:
        print("Failed to render the video. Skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
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
    args = parser.parse_args()

    robot = Robot(args.robot)

    env_cfg: WalkCfg | SquatCfg | RotateTorsoCfg | BalanceCfg | None = None
    train_cfg: PPOConfig | None = None
    EnvClass: (
        Type[WalkEnv] | Type[SquatEnv] | Type[RotateTorsoEnv] | Type[BalanceEnv] | None
    ) = None

    if "walk" in args.env:
        env_cfg = WalkCfg()
        train_cfg = PPOConfig()
        EnvClass = WalkEnv
        fixed_command = jnp.array([0.1, 0.0, 0.0])
        kwargs = {"ref_motion_type": "zmp"}

    elif "squat" in args.env:
        env_cfg = SquatCfg()
        train_cfg = PPOConfig()
        EnvClass = SquatEnv
        fixed_command = jnp.array([-0.0])
        kwargs = {}

    elif "rotate_torso" in args.env:
        env_cfg = RotateTorsoCfg()
        train_cfg = PPOConfig()
        EnvClass = RotateTorsoEnv
        fixed_command = jnp.array([0.2, 0.0])
        kwargs = {}

    elif "balance" in args.env:
        env_cfg = BalanceCfg()
        train_cfg = PPOConfig()
        EnvClass = BalanceEnv
        fixed_command = jnp.array([0.0])
        kwargs = {}

    else:
        raise ValueError(f"Unknown env: {args.env}")

    if "fixed" in args.env:
        train_cfg.num_timesteps = 10_000_000
        train_cfg.num_evals = 100

        env_cfg.rewards.healthy_z_range = [-0.2, 0.2]
        env_cfg.rewards.scales.reset()

        if "walk" in args.env:
            env_cfg.rewards.scales.feet_distance = 0.5

        env_cfg.rewards.scales.leg_joint_pos = 5.0
        env_cfg.rewards.scales.waist_joint_pos = 5.0
        env_cfg.rewards.scales.motor_torque = 5e-2
        env_cfg.rewards.scales.joint_acc = 5e-6
        env_cfg.rewards.scales.leg_action_rate = 1e-2
        env_cfg.rewards.scales.leg_action_acc = 1e-2
        env_cfg.rewards.scales.waist_action_rate = 1e-2
        env_cfg.rewards.scales.waist_action_acc = 1e-2

    env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        **kwargs,  # type: ignore
    )
    eval_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        **kwargs,  # type: ignore
    )
    test_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        fixed_command=fixed_command,
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

    run_name = f"{robot.name}_{args.env}_ppo_{time_str}"

    if len(args.eval) > 0:
        if os.path.exists(os.path.join("results", run_name)):
            evaluate(test_env, make_networks_factory, run_name)
        else:
            raise FileNotFoundError(f"Run {args.eval} not found.")
    else:
        train(env, eval_env, make_networks_factory, train_cfg, run_name, args.restore)

        evaluate(test_env, make_networks_factory, run_name)
