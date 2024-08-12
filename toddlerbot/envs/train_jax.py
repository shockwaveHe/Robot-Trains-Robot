import argparse
import functools
import os
import time
from typing import Any

from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
from brax.training.agents.ppo import train as ppo  # type: ignore
from flax.training import orbax_utils
from orbax import checkpoint as ocp  # type: ignore

import wandb
from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.envs.mujoco_env import MuJoCoEnv
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot


def train(env: MuJoCoEnv, train_cfg: PPOConfig, exp_folder_path: str):
    # # define the jit reset/step functions
    # jit_reset = jax.jit(env.reset)  # type: ignore
    # jit_step = jax.jit(env.step)  # type: ignore
    # jit_reset = env.reset
    # jit_step = env.step

    # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    # initialize the state
    # state = jit_reset(jax.random.PRNGKey(0))  # type: ignore
    # rollout: List[State] = [state.pipeline_state]  # type: ignore

    # grab a trajectory
    # for _ in tqdm(range(1000), desc="Simulating"):
    #     ctrl = -0.1 * jnp.ones(env.sys.nu)  # type: ignore
    #     state = jit_step(state, ctrl)  # type: ignore
    #     rollout.append(state.pipeline_state)  # type: ignore

    # profiler.disable()

    # stats = pstats.Stats(profiler).sort_stats(SortKey.TIME)
    # stats.print_stats(10)  # Print

    # media.write_video("test.mp4", env.render(rollout, camera="side"), fps=1.0 / env.dt)  # type: ignore

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        # save checkpoints
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.join(exp_folder_path, f"{current_step}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)  # type: ignore

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
    )

    train_fn = functools.partial(  # type: ignore
        ppo.train,
        network_factory=make_networks_factory,  # type: ignore
        # randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        **train_cfg.__dict__,
    )

    times = [time.time()]

    def progress(num_steps: int, metrics: Any):
        print(f"Step: {num_steps}, Reward: {metrics['eval/episode_reward']:.3f}")

        times.append(time.time())

        # Log metrics to wandb
        wandb.log(  # type: ignore
            {
                **metrics,
                "num_steps": num_steps,
                "time_elapsed": times[-1] - times[0],
            }
        )

    train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


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

    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name: str = f"{robot.name}_{motion_ref.name}_ppo_{time_str}"
    exp_folder_path = os.path.abspath(os.path.join("results", run_name))
    os.makedirs(exp_folder_path, exist_ok=True)

    wandb.init(  # type: ignore
        project="ToddlerBot",
        sync_tensorboard=True,
        name=run_name,
        config=train_cfg.__dict__,
    )

    train(env, train_cfg, exp_folder_path)
