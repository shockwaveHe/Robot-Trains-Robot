import time
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo

from toddlerbot.finetuning.finetune_config import FinetuneConfig, get_finetune_config
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer, RemoteReplayBuffer
from toddlerbot.finetuning.server_client import RemoteClient
from toddlerbot.finetuning.utils import Timer
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from toddlerbot.reference.balance_pd_ref import BalancePDReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import (
    exponential_moving_average,
    interpolate_action,
    inverse_exponential_moving_average,
)

# from toddlerbot.utils.misc_utils import profile
torch._dynamo.config.suppress_errors = True


class SwingPolicy(MJXFinetunePolicy, policy_name="swing"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpts: List[str],
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        exp_folder: Optional[str] = "",
        env_cfg: Optional[Dict] = None,
        finetune_cfg: Optional[Dict] = None,
        is_real: bool = True,
        ip: Optional[str] = None,
        eval_mode: bool = False,
    ):
        self.eval_mode = eval_mode
        if env_cfg is None:
            env_cfg = get_env_config("swing")
        if finetune_cfg is None:
            finetune_cfg = get_finetune_config("swing", exp_folder)

        # import ipdb; ipdb.set_trace()

        self.finetune_cfg: FinetuneConfig = finetune_cfg

        self.timer = Timer()
        self.num_privileged_obs_history = self.finetune_cfg.frame_stack
        self.privileged_obs_size = self.finetune_cfg.num_single_privileged_obs
        self.privileged_obs_history_size = (
            self.privileged_obs_size * self.num_privileged_obs_history
        )
        self.swing_buffer_size = self.finetune_cfg.swing_buffer_size
        self.fx_buffer = deque(maxlen=self.swing_buffer_size)
        self.fy_buffer = deque(maxlen=self.swing_buffer_size)
        self.fz_buffer = deque(maxlen=self.swing_buffer_size)
        self.pitch_buffer = deque(maxlen=self.swing_buffer_size)
        self.time_buffer = deque(maxlen=self.swing_buffer_size)

        # only call the init method of the grandparent MJXPolicy
        super(MJXFinetunePolicy, self).__init__(
            name,
            robot,
            init_motor_pos,
            "",
            joystick,
            fixed_command,
            env_cfg,
            exp_folder=exp_folder,
            need_warmup=False,
        )
        # self.control_dt = 0.1
        self.robot = robot
        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.finetune_cfg.update_mode == "local"
            else "cpu"
        )
        self.inference_device = "cpu"
        self.rng = np.random.default_rng()
        self.num_obs_history = self.cfg.obs.frame_stack
        self.obs_size = self.finetune_cfg.num_single_obs

        self.is_real = is_real
        self.is_paused = False
        self.active_motor_idx = [4, 7, 9, 10, 13, 15]
        self.active_motor_names = [
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
        ]
        self.action_delta_limit = 5 * self.control_dt
        self.action_mask = np.array(self.active_motor_idx)
        self.num_active_motors = self.action_mask.shape[0]
        self.action_deltas = deque(maxlen=self.finetune_cfg.action_window_size)
        self.vel_deltas = deque(maxlen=self.finetune_cfg.action_window_size)
        if self.finetune_cfg.swing_squat:
            self.motion_ref = BalancePDReference(robot, self.control_dt)
            self.com_ik_indices = [x - 4 for x in self.active_motor_idx]
            self.num_action = 1
            self.default_action = -0.04
            self.action_scale = 1.0
        else:
            self.num_action = self.num_active_motors
            if self.finetune_cfg.symmetric_action:
                self.num_action //= 2

            self.default_action = np.zeros(self.num_active_motors, dtype=np.float32)
            # self.default_action = self.default_motor_pos[self.action_mask]

        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_action_target = self.default_action
        self.last_raw_action = None
        self.is_stopped = False
        self.action_shift_steps = 1

        if self.finetune_cfg.update_mode == "local":
            self.replay_buffer = OnlineReplayBuffer(
                self.device,
                self.obs_size * self.num_obs_history,
                self.privileged_obs_size * self.num_privileged_obs_history,
                self.num_action,
                self.finetune_cfg.buffer_size,
                enlarge_when_full=self.finetune_cfg.update_interval
                * self.finetune_cfg.enlarge_when_full,
            )
            self.remote_client = None
        else:
            assert self.finetune_cfg.update_mode == "remote"
            self.remote_client = RemoteClient(
                # server_ip='192.168.0.227',
                server_ip="172.24.68.176",
                server_port=5007,
                exp_folder=self.exp_folder,
            )
            self.replay_buffer = RemoteReplayBuffer(
                self.remote_client,
                self.finetune_cfg.buffer_size,
                num_obs_history=self.num_obs_history,
                num_privileged_obs_history=self.num_privileged_obs_history,
                enlarge_when_full=self.finetune_cfg.update_interval
                * self.finetune_cfg.enlarge_when_full,
            )

        self._make_networks(
            observation_size=self.finetune_cfg.frame_stack * self.obs_size,
            privileged_observation_size=self.finetune_cfg.frame_stack
            * self.privileged_obs_size,
            action_size=self.num_action,
            value_hidden_layer_sizes=self.finetune_cfg.value_hidden_layer_sizes,
            policy_hidden_layer_sizes=self.finetune_cfg.policy_hidden_layer_sizes,
        )
        # self.load_networks(
        #     "results/toddlerbot_2xm_swing_real_world_20250426_153420",
        #     data_only=False,
        #     suffix="_final",
        # )

        if self.finetune_cfg.use_residual:
            self._make_residual_policy()
            self._residual_action_scale = self.finetune_cfg.residual_action_scale

        self.obs_history = np.zeros(
            self.num_obs_history * self.obs_size, dtype=np.float32
        )
        self.obs_history_size = self.num_obs_history * self.obs_size
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size, dtype=np.float32
        )

        if self.is_real:
            self.zmq_receiver = ZMQNode(type="receiver")
            assert len(ip) > 0, "Please provide the IP address of the sender"
            self.zmq_sender = ZMQNode(type="sender", ip=ip)
        else:
            self.zmq_receiver = None
            self.zmq_sender = None

        self.logger = FinetuneLogger(self.exp_folder)

        self.num_updates = 0
        self.total_steps, self.current_steps = 0, 0

        self.external_guidance_stage = "free"
        # self.step_limit = int(5e4)

        self._init_reward()
        self._make_learners()

        self.need_reset = True
        self.learning_stage = "offline"

        if len(ckpts) > 0:
            self.load_ckpts(ckpts)

        # if is_real:
        #     input("press ENTER to start")

        # self.health_xy_force = 10
        # self.health_z_force = 40
        # self.freq_tolerance = 1.0
        # self.amplitude = 0.8
        # self.period = 1.5

        self.m = 3.6
        g = 9.81
        theta_0 = np.pi / 6  # maximum angle (radians)
        L = 0.55
        # Under small-angle assumptions:
        self.desired_fx_amp = self.m * g * theta_0
        self.desired_fx_freq = np.sqrt(g / L) / (2 * np.pi)
        self.cycle_time = 1 / self.desired_fx_freq
        self.phase_signal = np.zeros(2, dtype=np.float32)

        self.reward_list = []
        self.reward_epi_list = []
        self.reward_epi_best = -np.inf

        self.ang_vel_z_max = 0.0
        self.fx_max = 0.0
        self.fx_amp_max = 0.0

        self.last_msg = None
        self.total_train_steps = 25000

    def get_obs(
        self, obs: Obs, command: np.ndarray = None, phase_signal=None, last_action=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        obs_arr = np.concatenate(
            [
                self.phase_signal,  # (2, )
                # command[self.command_obs_indices],  # (3, )
                motor_pos_delta[self.action_mask] * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel[self.action_mask] * self.obs_scales.dof_vel,  # (30, )
                self.last_action,
                obs.ang_vel * self.obs_scales.ang_vel,  # (3, )
                obs.euler * self.obs_scales.euler,  # (3, )
            ]
        )
        privileged_obs_arr = np.concatenate(
            [
                self.phase_signal,  # (2, )
                # command[self.command_obs_indices],  # (3, )
                motor_pos_delta[self.action_mask] * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel[self.action_mask] * self.obs_scales.dof_vel,  # (30, )
                self.last_action,
                obs.ang_vel * self.obs_scales.ang_vel,  # (3, )
                obs.euler * self.obs_scales.euler,  # (3, )
                obs.ee_force * self.obs_scales.ee_force,  # (3, )
                obs.ee_torque * self.obs_scales.ee_torque,  # (3, )
                obs.arm_ee_pos * self.obs_scales.arm_ee_pos,  # (3, )
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)
        self.obs_history[: obs_arr.size] = obs_arr
        self.privileged_obs_history = np.roll(
            self.privileged_obs_history, privileged_obs_arr.size
        )
        self.privileged_obs_history[: privileged_obs_arr.size] = privileged_obs_arr
        return self.obs_history, self.privileged_obs_history

    def get_raw_action(self, obs: Obs) -> np.ndarray:
        motor_target = obs.motor_pos.copy()
        action_target = motor_target[self.action_mask]

        if self.filter_type == "ema":
            action_target = inverse_exponential_moving_average(
                self.ema_alpha, action_target, self.last_raw_action
            )
            self.last_raw_action = action_target

        if self.finetune_cfg.swing_squat:
            knee_angle, hip_pitch_angle = action_target[1], action_target[0]
            com_pos = self.motion_ref.com_fk(knee_angle, hip_pitch_angle)
            action_target = np.array([com_pos[2]])

        raw_action = (action_target - self.default_action) / self.action_scale
        return raw_action

    def get_phase_signal(self, time_curr: float):
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    # @profile()
    def step(self, obs: Obs, is_real: bool = True):
        if self.need_reset:
            self.reset(obs)
            self.need_reset = False

        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 5.0 if is_real else 0.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                obs.motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=2.0 if is_real else 0.0,
            )

        msg = None
        while self.is_real and msg is None:
            msg = self.zmq_receiver.get_msg()

        # msg = self.zmq_receiver.get_msg()
        # if msg is None:
        #     msg = self.last_msg

        # self.last_msg = msg
        self.external_guidance_stage = msg.external_guidance_stage
        if msg.is_paused and not self.is_paused:
            self.timer.stop()
            self.is_paused = True
            print("Paused!")
        elif not msg.is_paused and self.is_paused:
            self.need_reset = True
            self.is_prepared = False
            self.is_paused = False
            print("Resumed!")
            return {}, obs.motor_pos, obs
        if self.is_paused:
            return {}, obs.motor_pos, obs

        # if msg.is_stopped:
        #     self.stopped = True
        #     print("Stopped!")
        #     return {}, np.zeros(self.num_action), obs

        obs.ee_force = msg.arm_force
        obs.ee_torque = msg.arm_torque
        obs.arm_ee_pos = msg.arm_ee_pos
        obs.arm_ee_vel = msg.arm_ee_vel

        self.fx_buffer.append(obs.ee_force[0])
        self.fy_buffer.append(obs.ee_force[1])
        self.fz_buffer.append(obs.ee_force[2])

        cur_time = time.time()
        if cur_time - self.traj_start_time < self.prep_duration:
            motor_target = np.asarray(
                interpolate_action(
                    cur_time - self.traj_start_time, self.prep_time, self.prep_action
                )
            )
            self.last_action_target = motor_target[self.action_mask]
            return {}, motor_target, obs

        n_buffer = len(self.fx_buffer)
        fx_fft_vals = np.fft.rfft(self.fx_buffer)
        fx_fft_freqs = np.fft.rfftfreq(n_buffer, d=self.control_dt)
        # fx_fft_phase = np.angle(fx_fft_vals)
        # Normalize the amplitude spectrum
        fx_amplitudes = np.abs(fx_fft_vals) / n_buffer * 2
        # Extract the amplitude at the desired frequency by finding the closest bin.
        # fx_freq_idx = np.argmin(np.abs(fx_fft_freqs - self.desired_fx_freq))

        # fx_freq_idx = np.argmax(fx_fft_vals[1:]) + 1  # Skip DC component
        fx_freq_idx = np.argmax(np.abs(fx_fft_vals[1:])) + 1  # Skip DC component
        self.fx_freq = fx_fft_freqs[fx_freq_idx]
        self.fx_amp = fx_amplitudes[fx_freq_idx]

        if self.fx_amp > self.fx_amp_max:
            self.fx_amp_max = self.fx_amp

        if abs(obs.ee_force[0]) > self.fx_max:
            self.fx_max = abs(obs.ee_force[0])

        if abs(obs.ang_vel[2]) > self.ang_vel_z_max:
            self.ang_vel_z_max = abs(obs.ang_vel[2])

        # print(f"fx_freq: {self.fx_freq}, fx_amp: {self.fx_amp}")
        # print(len(self.fx_buffer))

        time_curr = self.step_curr * self.control_dt
        self.phase_signal = self.get_phase_signal(time_curr)
        obs_arr, privileged_obs_arr = self.get_obs(obs)
        if self.total_steps == self.finetune_cfg.offline_total_steps:
            self.switch_learning_stage()
            if len(self.replay_buffer) > 0:
                self.replay_buffer.shift_action(
                    self.action_shift_steps
                )  # TODO: only support continuous data collection, no offline updates in between

        if self.remote_client is not None and self.remote_client.ready_to_update:
            # import ipdb; ipdb.set_trace()
            self.num_updates += 1
            print(f"Updated policy network to {self.num_updates}!")
            assert not torch.allclose(
                self.policy_net.mlp.layers[0].weight,
                self.remote_client.new_state_dict["mlp.layers.0.weight"].to(
                    self.inference_device
                ),
            )
            self.policy_net.load_state_dict(self.remote_client.new_state_dict)
            self.remote_client.ready_to_update = False
            self.need_reset = True

            # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
            motor_target = self.default_motor_pos.copy()
            motor_target[self.action_mask] = self.last_action_target
            return {}, motor_target, obs

        if (
            self.learning_stage == "online"
        ):  # use deterministic action during offline learning
            action_pi, action_real, action_logprob = self.get_action(
                obs_arr, deterministic=self.eval_mode, is_real=is_real
            )
            action_real_copy = action_real.copy()
        else:
            # data collection stage
            # action_pi = self.get_raw_action(obs)
            # action_logprob = 0.0
            action_pi, action_real, action_logprob = self.get_action(
                obs_arr, deterministic=self.eval_mode, is_real=is_real
            )
            action_real_copy = action_real.copy()

        if self.finetune_cfg.update_mode == "local":
            reward_dict = self._compute_reward(obs, action_real)
            self.last_last_action = self.last_action.copy()
            self.last_action = action_pi.copy()

            reward = sum(reward_dict.values())  # TODO: verify, why multiply by dt?
        else:
            reward = 0.0

        self.reward_list.append(reward)

        # print(reward_dict)
        time_elapsed = self.timer.elapsed()
        if time_elapsed < self.total_steps * self.control_dt:
            time.sleep(self.total_steps * self.control_dt - time_elapsed)

        if (len(self.replay_buffer) + 1) % 200 == 0:
            print(
                f"Data size: {len(self.replay_buffer)}, Steps: {self.total_steps}, Fps: {self.total_steps / self.timer.elapsed()}"
            )
            self.send_step_count()

        time_to_update = (
            #     not self.eval_mode
            #     and self.learning_stage == "offline"
            #     and (
            #         len(self.replay_buffer) >= self.finetune_cfg.offline_initial_steps
            #         and len(self.replay_buffer) + 1
            #     )
            #     % self.finetune_cfg.update_interval
            #     == 0
            # ) or (
            not self.eval_mode
            and self.learning_stage == "online"
            and len(self.replay_buffer) == self.finetune_cfg.online.batch_size - 1
        )
        truncated = (
            time_to_update
            and self.finetune_cfg.update_mode == "local"
            and self.total_steps < self.total_train_steps
        )

        # if self.is_truncated():
        #     truncated = True
        #     self.current_steps = 0
        #     print("Truncated! Resetting...")

        self.replay_buffer.store(
            obs_arr,
            privileged_obs_arr,
            action_pi,
            reward,
            msg.is_stopped,
            truncated or self.is_paused,
            action_logprob,
            raw_obs=deepcopy(obs),
        )

        if truncated:
            reward_epi = np.mean(self.reward_list)
            self.reward_epi_list.append(reward_epi)
            if (
                reward_epi > self.reward_epi_best
                and self.external_guidance_stage == "free"
            ):
                self.reward_epi_best = reward_epi
                self.save_networks(suffix="_best")
                print(f"Best reward: {self.reward_epi_best}")

            self.update_policy()
            self.need_reset = True
            # import ipdb; ipdb.set_trace()

        if not self.finetune_cfg.swing_squat and self.finetune_cfg.symmetric_action:
            action_real = np.concatenate([-action_real, action_real])

        if is_real:
            delayed_action = action_real
        else:
            self.action_buffer = np.roll(self.action_buffer, action_real.size)
            self.action_buffer[: action_real.size] = action_real
            if self.finetune_cfg.symmetric_action:
                delayed_action = self.action_buffer[-self.num_action * 2 :]
            else:
                delayed_action = self.action_buffer[-self.num_action :]
        # import ipdb; ipdb.set_trace()

        if self.finetune_cfg.swing_squat:
            com_z_target = self.default_action + self.action_scale * delayed_action[0]
            com_z_target = np.clip(
                com_z_target,
                self.motion_ref.com_z_limits[0],
                self.motion_ref.com_z_limits[1],
            )
            action_target = self.motion_ref.com_ik(com_z_target)[self.com_ik_indices]
        else:
            action_target = self.default_action + self.action_scale * delayed_action
            # lower = self.motor_limits[self.action_mask, 0]
            # upper = self.motor_limits[self.action_mask, 1]
            # action_target = 0.5 * (delayed_action + 1) * (upper - lower) + lower

        if self.filter_type == "ema":
            action_target = exponential_moving_average(
                self.ema_alpha, action_target, self.last_action_target
            )

        action_target = np.clip(
            action_target,
            self.last_action_target - self.action_delta_limit,
            self.last_action_target + self.action_delta_limit,
        )
        self.last_action_target = action_target.copy()

        # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
        motor_target = self.default_motor_pos.copy()
        motor_target[self.action_mask] = action_target

        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        if self.finetune_cfg.update_mode == "local":
            motor_angles = dict(zip(self.active_motor_names, action_target))
            self.logger.log_step(
                reward_dict,
                obs,
                reward=reward,
                action_pi=action_pi.mean(),
                action_real=action_real_copy.mean(),
                **motor_angles,
            )

        self.total_steps += 1
        self.current_steps += 1
        self.step_curr += 1
        self.timer.start()
        control_inputs = {}
        self.control_inputs = control_inputs

        return control_inputs, motor_target, obs

    def close(self):
        self.logger.close()
        save_networks = input("Save networks? y/n:")
        while save_networks not in ["y", "n"]:
            save_networks = input("Save networks? y/n:")
        if save_networks == "y":
            self.save_networks(suffix="_final")
        save_buffer = input("Save replay buffer? y/n:")
        if save_buffer == "y":
            self.replay_buffer.save_compressed(self.exp_folder)

    # def is_truncated(self) -> bool:
    #     return self.current_steps >= self.step_limit

    def reset(self, obs: Obs = None):
        # mjx policy reset
        self.timer.stop()
        self.step_curr = 0.0
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)
        self.privileged_obs_history = np.zeros(
            self.privileged_obs_history_size, dtype=np.float32
        )
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_action_target = self.default_action
        self.last_raw_action = None
        self.traj_start_time = time.time()
        self.is_prepared = False
        self.fx_buffer.clear()
        self.fy_buffer.clear()
        self.fz_buffer.clear()
        self.pitch_buffer.clear()
        self.time_buffer.clear()
        self.reward_list = []
        print("Reset done!")

    def _reward_survival(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # if self.is_done(obs):
        #     print("done reward")
        # return self.is_done(obs)
        return 1.0

    def _reward_torso_pitch(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Encourage large torso pitch deviations"""
        return np.clip(np.abs(obs.euler[1]), 0, 0.5)

    def _reward_fx_sine_amp(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for large Fx amplitude"""
        reward = np.exp(-0.005 * (self.fx_amp - self.desired_fx_amp) ** 2)
        return reward
        # return np.square(0.5 * self.fx_amp)

    def _reward_fx_sine_freq(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for good Fx sine fit"""
        reward = np.exp(-0.5 * (self.fx_freq - self.desired_fx_freq) ** 2)
        return reward

    def _reward_fx(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        return np.square(0.5 * obs.ee_force[0])

    def _reward_fz_sine_amp(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for appropriate Fz amplitude"""
        reward = np.exp(-0.005 * (self.fz_amp - self.desired_fz_amp) ** 2)
        return reward

    def _reward_fz_sine_freq(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for good Fz sine fit"""
        reward = np.exp(-0.5 * (self.fz_freq - self.desired_fz_freq) ** 2)
        return reward

    def _reward_action_rate(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - self.last_action)
        reward = -np.mean(error)
        return reward

    def _reward_action_acc(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - 2 * self.last_action + self.last_last_action)
        reward = -np.mean(error)
        return reward

    def _reward_action_symmetry(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        return -np.square(
            action[: self.num_action // 2] + action[self.num_action // 2 :]
        ).mean()

    def _reward_ang_vel_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        reward = np.exp(-0.01 * obs.ang_vel[2] ** 2)
        return reward
