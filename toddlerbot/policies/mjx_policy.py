import functools
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.ppo_config import PPOConfig
from toddlerbot.policies import BasePolicy
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import (
    butterworth,
    exponential_moving_average,
    interpolate_action,
)

# from toddlerbot.utils.misc_utils import profile


class MJXPolicy(BasePolicy, policy_name="mjx"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        cfg: Optional[MJXConfig] = None,
        motion_ref: Optional[MotionReference] = None,
    ):
        super().__init__(name, robot, init_motor_pos)

        assert cfg is not None, "cfg is required in the subclass!"
        assert motion_ref is not None, "motion_ref is required in the subclass!"

        if fixed_command is None:
            self.fixed_command = np.zeros(cfg.commands.num_commands, dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.motion_ref = motion_ref

        self.command_list = cfg.commands.command_list

        self.obs_scales = cfg.obs.scales  # Assume all the envs have the same scales
        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.action_scale = cfg.action.action_scale
        self.n_steps_delay = cfg.action.n_steps_delay

        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        self.neck_yaw_idx = robot.motor_ordering.index("neck_yaw_drive")
        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_drive")

        # Filter
        self.filter_type = cfg.action.filter_type
        self.filter_order = cfg.action.filter_order
        # EMA
        self.ema_alpha = float(
            cfg.action.filter_cutoff
            / (cfg.action.filter_cutoff + 1 / (self.control_dt * 2 * jnp.pi))
        )
        # Butterworth
        b, a = scipy.signal.butter(
            self.filter_order,
            cfg.action.filter_cutoff / (0.5 / self.control_dt),
            btype="low",
            analog=False,
        )
        self.butter_b_coef = np.array(b)[:, None]
        self.butter_a_coef = np.array(a)[:, None]

        self.last_motor_target = self.default_motor_pos.copy()
        self.butter_past_inputs = np.tile(
            self.last_motor_target, (self.filter_order, 1)
        )
        self.butter_past_outputs = np.tile(
            self.last_motor_target, (self.filter_order, 1)
        )

        self.state_ref = None
        self.last_action = np.zeros(robot.nu, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * robot.nu), dtype=np.float32
        )
        self.obs_history = np.zeros(
            cfg.obs.frame_stack * cfg.obs.num_single_obs, dtype=np.float32
        )
        self.step_curr = 0

        train_cfg = PPOConfig()
        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
        )

        ppo_network = make_networks_factory(
            cfg.obs.num_single_obs,
            cfg.obs.num_single_privileged_obs,
            robot.nu,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)

        if len(ckpt) > 0:
            run_name = f"{robot.name}_{self.name}_ppo_{ckpt}"
            policy_path = os.path.join("results", run_name, "best_policy")
            if not os.path.exists(policy_path):
                policy_path = os.path.join("results", run_name, "policy")
        else:
            policy_path = os.path.join(
                "toddlerbot",
                "policies",
                "checkpoints",
                f"{robot.name}_{self.name}_policy",
            )

        print(f"Loading policy from {policy_path}")

        params = model.load_params(policy_path)
        inference_fn = make_policy(params, deterministic=True)
        # jit_inference_fn = inference_fn
        self.jit_inference_fn = jax.jit(inference_fn)
        self.rng = jax.random.PRNGKey(0)
        self.jit_inference_fn(self.obs_history, self.rng)[0].block_until_ready()

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=5.0,
        )

    def is_double_support(self) -> bool:
        if self.state_ref is None:
            return False

        stance_mask = self.state_ref[-2:]
        return stance_mask[0] == 1.0 and stance_mask[1] == 1.0

    def get_command(self) -> npt.NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    # @profile()
    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        time_curr = self.step_curr * self.control_dt

        if self.joystick is None:
            command = self.fixed_command
        else:
            command = self.get_command()

        phase_signal = self.motion_ref.get_phase_signal(time_curr, command)
        self.state_ref = self.motion_ref.get_state_ref(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            time_curr,
            command,
        )
        motor_pos_delta = obs.motor_pos - self.default_motor_pos

        obs_arr = np.concatenate(
            [
                phase_signal,
                command,
                motor_pos_delta * self.obs_scales.dof_pos,
                obs.motor_vel * self.obs_scales.dof_vel,
                self.last_action,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)
        self.obs_history[: obs_arr.size] = obs_arr

        jit_action, _ = self.jit_inference_fn(jnp.asarray(self.obs_history), self.rng)

        action = np.asarray(jit_action, dtype=np.float32).copy()
        if is_real:
            action_delay = action
        else:
            self.action_buffer = np.roll(self.action_buffer, action.size)
            self.action_buffer[: action.size] = action
            action_delay = self.action_buffer[-self.robot.nu :]

        motor_target = self.default_motor_pos + self.action_scale * action_delay
        motor_target = np.asarray(
            self.motion_ref.override_motor_target(motor_target, self.state_ref)
        )

        if self.filter_type == "ema":
            motor_target = exponential_moving_average(
                self.ema_alpha, motor_target, self.last_motor_target
            )

        elif self.filter_type == "butter":
            (
                motor_target,
                self.butter_past_inputs,
                self.butter_past_outputs,
            ) = butterworth(
                self.butter_b_coef,
                self.butter_a_coef,
                motor_target,
                self.butter_past_inputs,
                self.butter_past_outputs,
            )

        # Keep the neck joints the same
        motor_target[self.neck_yaw_idx] = obs.motor_pos[self.neck_yaw_idx]
        motor_target[self.neck_pitch_idx] = obs.motor_pos[self.neck_pitch_idx]

        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.last_motor_target = motor_target.copy()
        self.last_action = action_delay
        self.step_curr += 1

        return motor_target
