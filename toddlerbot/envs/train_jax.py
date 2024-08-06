import functools
from datetime import datetime

import jax
from brax.training.agents.ppo import train as ppo
from matplotlib import pyplot as plt

from toddlerbot.envs.mujoco_env import MuJoCoEnv

env = MuJoCoEnv()

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# # initialize the state
# state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state.pipeline_state]

# # grab a trajectory
# for i in range(10):
#     ctrl = -0.1 * jp.ones(env.sys.nu)
#     state = jit_step(state, ctrl)
#     rollout.append(state.pipeline_state)


# media.write_video("test.mp4", env.render(rollout, camera="side"), fps=1.0 / env.dt)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=30_000_000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    seed=0,
)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    print(f"step: {num_steps}, reward: {metrics['eval/episode_reward']:.3f}")
    # plt.show()


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

plt.show()
