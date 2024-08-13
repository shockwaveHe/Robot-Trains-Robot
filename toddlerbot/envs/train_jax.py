import argparse
import functools
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import mediapy as media
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
from brax.training.agents.ppo import train as ppo  # type: ignore
from flax.training import orbax_utils
from orbax import checkpoint as ocp  # type: ignore
from tqdm import tqdm

import wandb
from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.envs.mujoco_env import MuJoCoEnv
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot


def log(
    metrics: Dict[str, Any],
    num_total_steps: int,
    num_steps: int,
    time_elapsed: float,
    width: int = 80,
    pad: int = 35,
):
    log_data: Dict[str, Any] = {"num_steps": num_steps, "time_elapsed": time_elapsed}
    title = f" \033[1m Learning steps {num_steps}/{num_total_steps } \033[0m "
    log_string = f"""{'#' * width}\n"""
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
        log_string += f"""{f'{metric_name}:':>{pad}} {value:.4f}\n"""

    log_string += (
        f"""{'Mean reward:':>{pad}} {metrics['eval/episode_reward']:.3f}\n"""
        f"""{'Mean episode length:':>{pad}} {metrics['eval/avg_episode_length']:.3f}\n"""
    )
    log_string += (
        f"""{'-' * width}\n""" f"""{'Time elapsed:':>{pad}} {time_elapsed:.1f}\n"""
    )
    if num_steps > 0:
        log_string += (
            f"""{'Computation:':>{pad}} {(num_steps / time_elapsed ):.1f} steps/s\n"""
            f"""{'ETA:':>{pad}} {(time_elapsed / num_steps) * (num_total_steps - num_steps):.1f}s\n"""
        )

    print(log_string)

    return log_data


def train(
    env: MuJoCoEnv, make_networks_factory: Any, train_cfg: PPOConfig, run_name: str
):
    exp_folder_path = os.path.join("results", run_name)
    os.makedirs(exp_folder_path, exist_ok=True)

    wandb.init(  # type: ignore
        project="ToddlerBot",
        sync_tensorboard=True,
        name=run_name,
        config=train_cfg.__dict__,
    )

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        # save checkpoints
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.abspath(os.path.join(exp_folder_path, f"{current_step}"))
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)  # type: ignore

    train_fn = functools.partial(  # type: ignore
        ppo.train,
        network_factory=make_networks_factory,  # type: ignore
        # randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        **train_cfg.__dict__,
    )

    times = [time.time()]

    def progress(num_steps: int, metrics: Dict[str, Any]):
        times.append(time.time())

        log_data = log(
            metrics, train_cfg.num_timesteps, num_steps, times[-1] - times[0]
        )

        # Log metrics to wandb
        wandb.log(log_data)  # type: ignore

    _, params, _ = train_fn(environment=env, progress_fn=progress)  # type: ignore

    model_path = os.path.join(exp_folder_path, "policy")
    model.save_params(model_path, params)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


def evaluate(
    env: MuJoCoEnv, make_networks_factory: Any, train_cfg: PPOConfig, run_name: str
):
    ppo_network = make_networks_factory(
        env.obs_size, env.privileged_obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(os.path.join("results", run_name, "policy"))
    inference_fn = make_policy(params)

    # initialize the state
    # jit_reset = env.reset
    # jit_step = env.step
    # jit_inference_fn = inference_fn
    jit_reset = jax.jit(env.reset)  # type: ignore
    jit_step = jax.jit(env.step)  # type: ignore
    jit_inference_fn = jax.jit(inference_fn)  # type: ignore

    rng = jax.random.PRNGKey(0)  # type: ignore
    command = jnp.array([1.0, 0.0, 0.0, 0.0])  # type: ignore
    state = jit_reset(rng)  # type: ignore
    state.info["command"] = command  # type: ignore
    rollout = [state.pipeline_state]  # type: ignore

    # grab a trajectory
    n_steps = 500
    render_every = 2

    for _ in tqdm(range(n_steps), desc="Evaluating"):
        act_rng, rng = jax.random.split(rng)  # type: ignore
        ctrl, _ = jit_inference_fn(state.obs, act_rng)  # type: ignore
        state = jit_step(state, ctrl)  # type: ignore
        rollout.append(state.pipeline_state)  # type: ignore

    media.write_video(
        os.path.join("results", run_name, "eval.mp4"),
        env.render(rollout[::render_every], camera="side"),  # type: ignore
        fps=1.0 / env.dt / render_every,
    )


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
        "--vis",
        type=str,
        default="",
        help="The name of the env.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    if args.env == "walk":
        from toddlerbot.motion_reference.walk_ref import WalkReference

        motion_ref = WalkReference(robot, use_jax=True)
    else:
        raise ValueError(f"Unknown env {args.env}")

    cfg = MuJoCoConfig()

    env = MuJoCoEnv(robot, motion_ref, cfg)

    train_cfg = PPOConfig()
    # train_cfg = PPOConfig(num_timesteps=100_000_000, num_evals=100)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    # time_str = "20240813_130046"
    run_name = f"{robot.name}_{motion_ref.name}_ppo_{time_str}"

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
    )

    train(env, make_networks_factory, train_cfg, run_name)

    evaluate(env, make_networks_factory, train_cfg, run_name)
