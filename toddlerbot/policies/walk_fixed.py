import functools
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore

from toddlerbot.envs.mjx_config import MuJoCoConfig
from toddlerbot.policies import BasePolicy
from toddlerbot.sim.robot import Robot


class WalkFixedPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "walk_fixed"

        cfg = MuJoCoConfig()
        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(128,) * 4,
            value_hidden_layer_sizes=(128,) * 4,
        )

        # joint indices
        motor_indices = np.arange(len(robot.motor_ordering))  # type:ignore
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

        self.last_action = jnp.zeros(len(robot.motor_ordering), dtype=jnp.float32)  # type:ignore
        self.obs_history = jnp.zeros(cfg.obs.frame_stack * cfg.obs.num_single_obs)  # type:ignore
        self.cycle_time = 1.2
        self.step = 0

        ppo_network = make_networks_factory(  # type: ignore
            cfg.obs.num_single_obs,
            cfg.obs.num_single_privileged_obs,
            len(robot.motor_ordering),
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)  # type: ignore
        policy_path = "results/toddlerbot_walk_fixed_ppo_20240819_111111/policy"
        params = model.load_params(policy_path)
        inference_fn = make_policy(params)
        # jit_inference_fn = inference_fn
        self.jit_inference_fn = jax.jit(inference_fn)  # type: ignore
        self.rng = jax.random.PRNGKey(0)  # type: ignore
        self.jit_inference_fn(self.obs_history, self.rng)[0].block_until_ready()  # type: ignore

    def run(
        self, obs_dict: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        phase = self.step * self.control_dt / self.cycle_time
        phase_signal = jnp.array(  # type:ignore
            [jnp.sin(2 * np.pi * phase), jnp.cos(2 * np.pi * phase)]  # type:ignore
        )
        command = jnp.zeros(3)  # type:ignore
        joint_pos_delta = obs_dict["q"] - self.default_joint_pos
        joint_vel = jnp.array(obs_dict["dq"])  # type:ignore
        torso_ang_vel = jnp.array(obs_dict["imu_ang_vel"])  # type:ignore
        torso_euler = jnp.array(obs_dict["imu_euler"])  # type:ignore
        obs = jnp.concatenate(  # type:ignore
            [
                phase_signal,
                command,
                joint_pos_delta * self.obs_scales.dof_pos,
                joint_vel * self.obs_scales.dof_vel,
                self.last_action,
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_euler * self.obs_scales.euler,
            ]
        )

        self.obs_history = jnp.roll(self.obs_history, obs.size).at[: obs.size].set(obs)  # type:ignore

        act_rng, self.rng = jax.random.split(self.rng)  # type: ignore
        jit_action, _ = self.jit_inference_fn(self.obs_history, act_rng)  # type: ignore
        jit_action = jnp.asarray(jit_action, dtype=jnp.float32)  # type:ignore

        zero_action = jnp.zeros_like(jit_action)  # type:ignore
        jit_action = jit_action.at[self.arm_motor_indices].set(  # type:ignore
            zero_action[self.arm_motor_indices]
        )
        jit_action = jit_action.at[self.neck_motor_indices].set(  # type:ignore
            zero_action[self.neck_motor_indices]
        )
        jit_action = jit_action.at[self.waist_motor_indices[-1]].set(  # type:ignore
            zero_action[self.waist_motor_indices[-1]]
        )

        self.last_action = jit_action
        self.step += 1

        action = (
            self.default_action
            + np.asarray(jit_action, dtype=np.float32) * self.action_scale
        )

        return action
