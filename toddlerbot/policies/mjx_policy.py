import functools
import os
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.teleop.joystick import get_controller_input, initialize_joystick
from toddlerbot.utils.math_utils import interpolate_action

# from toddlerbot.utils.misc_utils import profile


class RLPolicy(BasePolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        command_ranges: List[List[float]],
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        super().__init__(name, robot, init_motor_pos)

        self.command_ranges = command_ranges
        self.fixed_command = fixed_command

        train_cfg = PPOConfig()
        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
        )

        # joint indices
        motor_indices = np.arange(robot.nu)  # type:ignore
        motor_groups = np.array(
            [robot.joint_groups[name] for name in robot.motor_ordering]
        )
        self.leg_motor_indices = motor_indices[motor_groups == "leg"]
        self.arm_motor_indices = motor_indices[motor_groups == "arm"]
        self.neck_motor_indices = motor_indices[motor_groups == "neck"]
        self.waist_motor_indices = motor_indices[motor_groups == "waist"]

        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        cfg = MJXConfig()
        self.obs_scales = cfg.obs.scales  # Assume all the envs have the same scales
        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.action_scale = cfg.action.action_scale

        self.last_action = np.zeros(robot.nu, dtype=np.float32)
        self.obs_history = np.zeros(
            cfg.obs.frame_stack * cfg.obs.num_single_obs, dtype=np.float32
        )
        self.step_curr = 0

        ppo_network = make_networks_factory(  # type: ignore
            cfg.obs.num_single_obs,
            cfg.obs.num_single_privileged_obs,
            robot.nu,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)  # type: ignore

        run_name = f"{robot.name}_{self.name}_ppo_{ckpt}"
        policy_path = os.path.join("results", run_name, "best_policy")
        if not os.path.exists(policy_path):
            policy_path = os.path.join("results", run_name, "policy")

        params = model.load_params(policy_path)
        inference_fn = make_policy(params)
        # jit_inference_fn = inference_fn
        self.jit_inference_fn = jax.jit(inference_fn)  # type: ignore
        self.rng = jax.random.PRNGKey(0)  # type: ignore
        act_rng, _ = jax.random.split(self.rng)  # type: ignore
        self.jit_inference_fn(self.obs_history, act_rng)[0].block_until_ready()  # type: ignore

        self.joystick = initialize_joystick()

        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=5.0,
        )

    def get_phase_signal(self, time_curr: float) -> npt.NDArray[np.float32]:
        return np.zeros(2, dtype=np.float32)

    # @profile()
    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )

            return action

        time_curr = self.step_curr * self.control_dt
        phase_signal = self.get_phase_signal(time_curr)

        motor_pos_delta = obs.motor_pos - self.default_motor_pos

        if self.joystick is None:
            controller_input = self.fixed_command
        else:
            controller_input = get_controller_input(self.joystick, self.command_ranges)

        obs_arr = np.concatenate(  # type:ignore
            [
                phase_signal,
                np.array(controller_input),  # type:ignore
                motor_pos_delta * self.obs_scales.dof_pos,
                obs.motor_vel * self.obs_scales.dof_vel,
                self.last_action,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)  # type:ignore
        self.obs_history[: obs_arr.size] = obs_arr

        act_rng, self.rng = jax.random.split(self.rng)  # type: ignore
        jit_action, _ = self.jit_inference_fn(jnp.asarray(self.obs_history), act_rng)  # type: ignore

        action = np.asarray(jit_action, dtype=np.float32).copy()
        action[self.arm_motor_indices] = 0.0
        action[self.neck_motor_indices] = 0.0

        self.last_action = action
        self.step_curr += 1

        motor_target = np.where(  # type:ignore
            action < 0,
            self.default_motor_pos
            + self.action_scale
            * action
            * (self.default_motor_pos - self.motor_limits[:, 0]),
            self.default_motor_pos
            + self.action_scale
            * action
            * (self.motor_limits[:, 1] - self.default_motor_pos),
        )
        motor_target = np.clip(  # type:ignore
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        return motor_target
