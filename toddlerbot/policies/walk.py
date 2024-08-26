import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore

from toddlerbot.envs.mjx_config import MuJoCoConfig
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

# from toddlerbot.utils.misc_utils import profile


class WalkPolicy(BasePolicy):
    def __init__(self, robot: Robot, run_name: str):
        super().__init__(robot)
        self.name = "walk"

        cfg = MuJoCoConfig()
        train_cfg = PPOConfig()
        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
        )

        # joint indices
        motor_indices = np.arange(robot.action_size)  # type:ignore
        motor_groups = np.array(
            [robot.joint_group[name] for name in robot.motor_ordering]
        )
        self.leg_motor_indices = motor_indices[motor_groups == "leg"]
        self.arm_motor_indices = motor_indices[motor_groups == "arm"]
        self.neck_motor_indices = motor_indices[motor_groups == "neck"]
        self.waist_motor_indices = motor_indices[motor_groups == "waist"]

        self.action_scale = cfg.action.action_scale
        self.obs_scales = cfg.obs.scales
        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        self.last_action = np.zeros(robot.action_size, dtype=np.float32)
        self.obs_history = np.zeros(
            cfg.obs.frame_stack * cfg.obs.num_single_obs, dtype=np.float32
        )
        self.cycle_time = cfg.action.cycle_time
        self.step_curr = 0

        ppo_network = make_networks_factory(  # type: ignore
            cfg.obs.num_single_obs,
            cfg.obs.num_single_privileged_obs,
            robot.action_size,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)  # type: ignore

        policy_path = os.path.join("results", run_name, "policy")
        params = model.load_params(policy_path)
        inference_fn = make_policy(params)
        # jit_inference_fn = inference_fn
        self.jit_inference_fn = jax.jit(inference_fn)  # type: ignore
        self.rng = jax.random.PRNGKey(0)  # type: ignore
        self.jit_inference_fn(self.obs_history, self.rng)[0].block_until_ready()  # type: ignore

    # @profile()
    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        phase = self.step_curr * self.control_dt / self.cycle_time
        phase_signal = np.array(  # type:ignore
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]  # type:ignore
        )
        joint_pos_delta = obs.q - self.default_joint_pos
        obs_arr = np.concatenate(  # type:ignore
            [
                phase_signal,
                np.array([0.3, 0.0, 0.0]),  # type:ignore
                joint_pos_delta * self.obs_scales.dof_pos,
                obs.dq * self.obs_scales.dof_vel,
                self.last_action,
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)  # type:ignore
        self.obs_history[: obs_arr.size] = obs_arr

        # TODO: Remove after debugging
        act_rng, self.rng = jax.random.split(self.rng)  # type: ignore
        # act_rng = self.rng
        jit_action, _ = self.jit_inference_fn(jnp.asarray(self.obs_history), act_rng)  # type: ignore

        action = np.asarray(jit_action, dtype=np.float32).copy()
        action[self.arm_motor_indices] = 0.0
        action[self.neck_motor_indices] = 0.0
        action[self.waist_motor_indices[-1]] = 0.0

        self.last_action = action
        self.step_curr += 1

        motor_target = self.default_action + action * self.action_scale

        return motor_target
