import argparse
import functools
import importlib
import os
import pickle
import pkgutil
import time
from typing import Any, Dict, List

from brax import base

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["USE_JAX"] = "true"

import jax
import jax.numpy as jnp
import mediapy as media
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm

import wandb
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.locomotion.mjx_env import MJXEnv, get_env_class
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

    if "hang_force" in metrics:
        log_string += f"""{'Hang force:':>{pad}} {metrics['hang_force']:.3f}\n"""
    if "episode_num" in metrics:
        log_string += f"""{'Episode num:':>{pad}} {metrics['episode_num']}\n"""

    if num_steps > 0 and num_total_steps > 0:
        log_string += (
            f"""{'Computation:':>{pad}} {(num_steps / time_elapsed ):.1f} steps/s\n"""
            f"""{'ETA:':>{pad}} {(time_elapsed / num_steps) * (num_total_steps - num_steps):.1f}s\n"""
        )

    print(log_string)

    return log_data


def render_video(
    env: MJXEnv,
    rollout: List[Any],
    run_name: str,
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    # Render and save videos for each camera
    video_path = os.path.join("results", run_name, f"eval.mp4")
    print(f"Rendering video at {video_path}")
    media.write_video(
        video_path,
        env.render(
            rollout[::render_every],
            height=height,
            width=width,
            # eval=True,
        ),
        fps=1.0 / env.dt / render_every,
    )


def parse_domain_rand(sys: base.System, domain_rand_str: str):
    domain_rand_items = domain_rand_str.split(",")
    domain_rand_options = [
        "geom_friction",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "gravity",
    ]
    for domain_rand_item in domain_rand_items:
        domain_rand_key, domain_rand_val = domain_rand_item.split("=")
        if domain_rand_key not in domain_rand_options:
            raise ValueError(f"Invalid domain randomization option: {domain_rand_item}")
        geom_friction = (
            sys.geom_friction.at[:, 0].set(float(domain_rand_val))
            if domain_rand_key == "geom_friction"
            else sys.geom_friction
        )
        dof_damping = (
            sys.dof_damping.at[:].mul(float(domain_rand_val))
            if domain_rand_key == "dof_damping"
            else sys.dof_damping
        )
        dof_armature = (
            sys.dof_armature.at[:].mul(float(domain_rand_val))
            if domain_rand_key == "dof_armature"
            else sys.dof_armature
        )
        dof_frictionloss = (
            sys.dof_frictionloss.at[:].mul(float(domain_rand_val))
            if domain_rand_key == "dof_frictionloss"
            else sys.dof_frictionloss
        )
        gravity = (
            sys.opt.gravity.at[2].set(float(domain_rand_val))
            if domain_rand_key == "gravity"
            else sys.opt.gravity
        )
    new_opt = sys.opt.replace(gravity=gravity)
    sys_dict = {
        "geom_friction": geom_friction,
        "dof_damping": dof_damping,
        "dof_armature": dof_armature,
        "dof_frictionloss": dof_frictionloss,
        "opt": new_opt,
    }
    return sys.tree_replace(sys_dict)


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
    policy_path = os.path.join("toddlerbot", "policies", "checkpoints", "toddlerbot_walk_policy")
    # policy_path = os.path.join("results", run_name, "best_policy")
    # # policy_path = os.path.join("toddlerbot", "policies", "checkpoints", "toddlerbot_walk_policy")
    # if not os.path.exists(policy_path):
    #     policy_path = os.path.join("results", run_name, "policy")

    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(env.step)
    # jit_step = env.step
    jit_inference_fn = jax.jit(inference_fn)
    # jit_inference_fn = inference_fn

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    times = [time.time()]

    rollout: List[Any] = [state.pipeline_state]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        ctrl, _ = jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        times.append(time.time())
        rollout.append(state.pipeline_state)
        if i % log_every == 0:
            log_metrics(state.metrics, times[-1] - times[0])

    return rollout


def evaluate_batch(
    env: MJXEnv,
    make_networks_factory: Any,
    run_name: str,
    num_steps: int = 1000,
    log_every: int = 100,
    batch_size: int = 32,  # New parameter for batch size
):
    ppo_network = make_networks_factory(
        env.obs_size, #  * env.num_obs_history? 
        env.privileged_obs_size, #  * env.num_privileged_obs_history 
        env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_path = os.path.join("toddlerbot", "policies", "checkpoints", "toddlerbot_walk_policy")
    # policy_path = os.path.join("results", run_name, "best_policy")
    # if not os.path.exists(policy_path):
        # policy_path = os.path.join("results", run_name, "policy")

    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # Initialize the state
    # Vectorize reset and step functions
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    jit_inference_fn = jax.jit(jax.vmap(inference_fn))

    # Generate a batch of random keys
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, batch_size)

    state = jit_reset(rngs)

    times = [time.time()]
    rollout: List[Any] = [
        jax.device_get(state.pipeline_state)
    ]  # Now contains a batch of states

    for i in tqdm(range(num_steps), desc="Evaluating"):
        ctrl, _ = jit_inference_fn(state.obs, rngs)
        state = jit_step(state, ctrl)
        times.append(time.time())
        rollout.append(jax.device_get(state.pipeline_state))

        if i % log_every == 0:
            # Aggregate metrics across the batch
            avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), state.metrics)
            log_metrics(avg_metrics, times[-1] - times[0])
    # import ipdb
    # ipdb.set_trace()

    return rollout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--domain_rand",
        type=str,
        default="",
        help="The domain randomization to apply. Allowed keys: ['geom_friction': 0.5, 2.0, 'dof_damping': 0.8, 1.2, 'dof_armature': 0.8, 1.2, 'dof_frictionloss': 0.8, 1.2, 'gravity']",
    )
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    EnvClass = get_env_class(args.env)
    eval_cfg = get_env_config(args.env)
    eval_cfg.hang.init_hang_force = 0.0

    train_cfg = PPOConfig()
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )
    robot = Robot(args.robot)
    kwargs = {}
    if len(args.ref) > 0:
        kwargs = {"ref_motion_type": args.ref}

    eval_env = EnvClass(
        args.env,
        robot,
        eval_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        **kwargs,  # type: ignore
    )
    if len(args.domain_rand) > 0:
        assert isinstance(eval_env, MJXEnv)
        eval_env.sys = parse_domain_rand(eval_env.sys, args.domain_rand)
    run_name = f"{robot.name}_{args.env}_ppo_{args.run_name}"

    if args.batch_size > 1:
        rollout = evaluate_batch(
            eval_env, make_networks_factory, run_name, batch_size=args.batch_size
        )
    else:
        rollout = evaluate(eval_env, make_networks_factory, run_name)
    render_video(eval_env, rollout, run_name)
