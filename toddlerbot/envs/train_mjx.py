import os

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["USE_JAX"] = "true"

import argparse
import functools
import json
import shutil
import time
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import mediapy as media
import optax  # type: ignore
from brax import base  # type: ignore
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
from brax.training.agents.ppo import train as ppo  # type: ignore
from flax.training import orbax_utils
from moviepy.editor import VideoFileClip, clips_array  # type: ignore
from orbax import checkpoint as ocp  # type: ignore
from tqdm import tqdm

import wandb
from toddlerbot.envs.mjx_config import MJXConfig, RewardScales, RewardsConfig
from toddlerbot.envs.mjx_env import MuJoCoEnv
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot


def render_video(
    env: MuJoCoEnv,
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
            env.render(  # type: ignore
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


def domain_randomize(
    sys: base.System,
    rng: jax.Array,
    friction_range: Optional[List[float]],
    gain_range: Optional[List[float]],
    damping_range: Optional[List[float]],
    armature_range: Optional[List[float]],
) -> Tuple[base.System, base.System]:
    @jax.vmap
    def rand(rng: jax.Array) -> Tuple[jax.Array, ...]:
        _, key = jax.random.split(rng, 2)  # type: ignore

        if friction_range is None:
            friction = sys.geom_friction
        else:
            # Friction
            friction = jax.random.uniform(  # type: ignore
                key, (1,), minval=friction_range[0], maxval=friction_range[1]
            )  # type: ignore
            friction = sys.geom_friction.at[:, 0].set(friction)  # type: ignore

        if gain_range is None:
            gain = sys.actuator_gainprm
            bias = sys.actuator_biasprm
        else:
            # Actuator
            _, key = jax.random.split(key, 2)  # type: ignore
            param = (
                jax.random.uniform(  # type: ignore
                    key, (1,), minval=gain_range[0], maxval=gain_range[1]
                )
                + sys.actuator_gainprm[:, 0]
            )
            gain = sys.actuator_gainprm.at[:, 0].set(param)  # type: ignore
            bias = sys.actuator_biasprm.at[:, 1].set(-param)  # type: ignore

        if damping_range is None:
            damping = sys.dof_damping
        else:
            damping = (
                jax.random.uniform(  # type: ignore
                    key, (sys.nv,), minval=damping_range[0], maxval=damping_range[1]
                )
                * sys.dof_damping
            )

        if armature_range is None:
            armature = sys.dof_armature
        else:
            armature = (
                jax.random.uniform(  # type: ignore
                    key, (sys.nv,), minval=armature_range[0], maxval=armature_range[1]
                )
                * sys.dof_armature
            )

        return friction, gain, bias, damping, armature

    friction, gain, bias, damping, armature = rand(rng)

    in_axes = jax.tree.map(lambda x: None, sys)  # type: ignore
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "dof_damping": 0,
            "dof_armature": 0,
        }
    )

    sys = sys.tree_replace(  # type: ignore
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
            "dof_damping": damping,
            "dof_armature": armature,
        }
    )

    return sys, in_axes


def train(
    env: MuJoCoEnv,
    eval_env: MuJoCoEnv,
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

    wandb.init(  # type: ignore
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
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)  # type: ignore
        policy_path = os.path.join(path, "policy")
        model.save_params(policy_path, (params[0], params[1].policy))

    learning_rate_schedule_fn = optax.linear_schedule(  # type: ignore
        init_value=train_cfg.learning_rate,
        end_value=train_cfg.min_learning_rate,
        transition_steps=train_cfg.num_timesteps,
    )

    domain_randomize_fn = functools.partial(
        domain_randomize,
        friction_range=cfg.domain_rand.friction_range,
        gain_range=cfg.domain_rand.gain_range,
        damping_range=cfg.domain_rand.damping_range,
        armature_range=cfg.domain_rand.armature_range,
    )

    train_fn = functools.partial(  # type: ignore
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
        network_factory=make_networks_factory,  # type: ignore
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
        wandb.log(log_data)  # type: ignore

    _, params, _ = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)  # type: ignore

    model_path = os.path.join(exp_folder_path, "policy")
    model.save_params(model_path, params)

    shutil.copy2(
        os.path.join(exp_folder_path, str(best_ckpt_step), "policy"),
        os.path.join(exp_folder_path, "best_policy"),
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


def evaluate(
    env: MuJoCoEnv,
    make_networks_factory: Any,
    run_name: str,
    num_steps: int = 1000,
    log_every: int = 100,
):
    ppo_network = make_networks_factory(
        env.obs_size, env.privileged_obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)  # type: ignore
    policy_path = os.path.join("results", run_name, "best_policy")
    if not os.path.exists(policy_path):
        policy_path = os.path.join("results", run_name, "policy")

    params = model.load_params(policy_path)
    inference_fn = make_policy(params)

    # initialize the state
    # jit_reset = env.reset
    # jit_step = env.step
    # jit_inference_fn = inference_fn
    jit_reset = jax.jit(env.reset)  # type: ignore
    jit_step = jax.jit(env.step)  # type: ignore
    jit_inference_fn = jax.jit(inference_fn)  # type: ignore

    rng = jax.random.PRNGKey(0)  # type: ignore
    state = jit_reset(rng)  # type: ignore

    rollout: List[Any] = [state.pipeline_state]  # type: ignore

    times = [time.time()]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        act_rng, rng = jax.random.split(rng)  # type: ignore
        ctrl, _ = jit_inference_fn(state.obs, act_rng)  # type: ignore
        state = jit_step(state, ctrl)  # type: ignore
        times.append(time.time())
        rollout.append(state.pipeline_state)  # type: ignore
        if i % log_every == 0:
            log_metrics(state.metrics, times[-1] - times[0])  # type: ignore

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
        "--ref",
        type=str,
        default="walk_zmp",
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

    if args.env == "walk":
        cfg = MJXConfig()
        train_cfg = PPOConfig()
        # test_command = jnp.array([0.0, 0.0, 0.0])  # type:ignore
        test_command = jnp.array([0.3, 0.0, 0.0])  # type:ignore

    elif args.env == "walk_fixed":
        reward_scales = replace(
            RewardScales(),
            **{field: 0.0 for field in RewardScales.__dataclass_fields__},
        )
        # reward_scales.feet_distance = 0.5
        reward_scales.leg_joint_pos = 5.0
        reward_scales.waist_joint_pos = 5.0
        reward_scales.joint_torque = 5e-2
        reward_scales.joint_acc = 5e-6
        reward_scales.leg_action_rate = 1e-2
        reward_scales.leg_action_acc = 1e-2

        cfg = MJXConfig(
            rewards=RewardsConfig(healthy_z_range=[-0.2, 0.2], scales=reward_scales)
        )
        train_cfg = PPOConfig(
            num_timesteps=10_000_000,
            num_evals=100,
            transition_steps=1_000_000,
            learning_rate=1e-4,
        )
        # test_command = jnp.array([0.0, 0.0, 0.0])  # type:ignore
        test_command = jnp.array([0.3, 0.0, 0.0])  # type:ignore

    else:
        raise ValueError(f"Unknown env: {args.env}")

    # Need to a separate env for evaluation, otherwise the domain randomization will cause tracer leak errors.
    env = MuJoCoEnv(
        args.env, cfg, robot, ref_motion_name=args.ref, fixed_base="fixed" in args.env
    )
    eval_env = MuJoCoEnv(
        args.env, cfg, robot, ref_motion_name=args.ref, fixed_base="fixed" in args.env
    )

    test_env = MuJoCoEnv(
        args.env,
        cfg,
        robot,
        ref_motion_name=args.ref,
        fixed_base="fixed" in args.env,
        fixed_command=test_command,
        add_noise=False,
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
