import argparse
import cProfile
import functools
import pstats
import time
from pstats import SortKey
from typing import Any, Dict, List

import jax
import mediapy as media
from brax.base import State  # type: ignore
from brax.training.agents.ppo import train as ppo  # type: ignore
from jax import numpy as jnp

from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.envs.mujoco_env import MuJoCoEnv
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import set_seed


def train(robot: Robot, motion_ref: MotionReference):
    time_start = time.time()

    cfg = MuJoCoConfig()
    env = MuJoCoEnv(robot, motion_ref, cfg)

    # # define the jit reset/step functions
    # jit_reset = jax.jit(env.reset)  # type: ignore
    # jit_step = jax.jit(env.step)  # type: ignore
    jit_reset = env.reset
    # jit_step = env.step

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # initialize the state
    state = jit_reset(jax.random.PRNGKey(0))  # type: ignore
    rollout: List[State] = [state.pipeline_state]  # type: ignore

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats(SortKey.TIME)
    stats.print_stats(10)  # Print

    # # grab a trajectory
    # for _ in range(10):
    #     ctrl = -0.1 * jnp.ones(env.sys.nu)  # type: ignore
    #     state = jit_step(state, ctrl)  # type: ignore
    #     rollout.append(state.pipeline_state)  # type: ignore

    time_end = time.time()

    print(f"Time elapsed: {time_end - time_start}")

    media.write_video("test.mp4", env.render(rollout, camera="side"), fps=1.0 / env.dt)  # type: ignore

    # train_fn = functools.partial(
    #     ppo.train,
    #     num_timesteps=30_000_000,
    #     num_evals=5,
    #     reward_scaling=0.1,
    #     episode_length=1000,
    #     normalize_observations=True,
    #     action_repeat=1,
    #     unroll_length=10,
    #     num_minibatches=32,
    #     num_updates_per_batch=8,
    #     discounting=0.97,
    #     learning_rate=3e-4,
    #     entropy_cost=1e-3,
    #     num_envs=2048,
    #     batch_size=1024,
    #     seed=0,
    # )

    # def progress(num_steps: int, metrics: Dict[str, Any]):
    #     print(f"step: {num_steps}, reward: {metrics['eval/episode_reward']:.3f}")
    #     # plt.show()

    # make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)


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

    set_seed(0)

    robot = Robot(args.robot)

    if args.env == "walk":
        from toddlerbot.motion_reference.walk_ref import WalkReference

        motion_ref = WalkReference(robot, use_jax=True)
    else:
        raise ValueError(f"Unknown env {args.env}")

    train(robot, motion_ref)
