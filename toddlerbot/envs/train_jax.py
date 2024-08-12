import argparse
import functools
import os
import time
from typing import Any, List

from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
from brax.training.agents.ppo import train as ppo  # type: ignore
from flax.training import orbax_utils
from orbax import checkpoint as ocp  # type: ignore

from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.envs.mujoco_env import MuJoCoEnv
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot


def train(robot: Robot, motion_ref: MotionReference):
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

    exp_name: str = "test_ref_motion"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"
    os.makedirs(exp_folder_path, exist_ok=True)

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
        num_timesteps=10_000_000,
        num_evals=10,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3.0e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        network_factory=make_networks_factory,  # type: ignore
        # randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=0,
    )

    x_data: List[int] = []
    y_data: List[float] = []
    ydataerr: List[float] = []
    times = [time.time()]

    cfg = MuJoCoConfig()
    env = MuJoCoEnv(robot, motion_ref, cfg)

    def progress(num_steps: int, metrics: Any):
        print(f"Step: {num_steps}, Reward: {metrics['eval/episode_reward']:.3f}")

        times.append(time.time())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])
        # plt.show()

    train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    import matplotlib.pyplot as plt

    plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])  # type: ignore
    plt.ylim([0, 13000])  # type: ignore

    plt.xlabel("# environment steps")  # type: ignore
    plt.ylabel("reward per episode")  # type: ignore
    plt.title(f"y={y_data[-1]:.3f}")  # type: ignore

    plt.errorbar(x_data, y_data, yerr=ydataerr)  # type: ignore
    plt.savefig(f"{exp_folder_path}/reward.png")  # type: ignore


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

    train(robot, motion_ref)
