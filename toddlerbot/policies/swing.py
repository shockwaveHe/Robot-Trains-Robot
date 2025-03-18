from collections import deque
from copy import deepcopy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from toddlerbot.finetuning.finetune_config import get_finetune_config, FinetuneConfig
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action
from tqdm import tqdm
from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer, RemoteReplayBuffer
from toddlerbot.finetuning.server_client import RemoteClient
from toddlerbot.finetuning.dynamics import DynamicsNetwork, BaseDynamics
from toddlerbot.finetuning.utils import Timer
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.finetuning.networks import load_jax_params, load_jax_params_into_pytorch
import toddlerbot.finetuning.networks as networks
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.utils.math_utils import interpolate_action, exponential_moving_average, inverse_exponential_moving_average
from toddlerbot.utils.comm_utils import ZMQNode, ZMQMessage
from toddlerbot.utils.misc_utils import log, profile
from toddlerbot.finetuning.logger import FinetuneLogger
from scipy.optimize import curve_fit
from scipy.fft import rfft


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
        ip: Optional[str] = None
    ):
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
            need_warmup=False
        )

        # self.control_dt = 0.1
        self.robot = robot
        self.device = "cuda" if torch.cuda.is_available() and self.finetune_cfg.update_mode == 'local' else "cpu"
        self.inference_device = "cpu"
        self.rng = np.random.default_rng()
        self.num_obs_history = self.cfg.obs.frame_stack
        self.obs_size = self.finetune_cfg.num_single_obs

        self.is_real = is_real

        self.active_motor_idx = [4, 7, 10, 13]
        self.motor_speed_limits = np.array([0.05])

        self.action_mask = np.array(self.active_motor_idx)
        self.num_action = self.action_mask.shape[0]
        self.default_action = self.default_motor_pos[self.action_mask]
        self.last_action = np.zeros(self.num_action)
        self.last_last_action = np.zeros(self.num_action)
        self.last_action_target = self.default_action
        self.last_raw_action = None
        self.is_stopped = False
        self.is_paused = False
        self.action_shift_steps = 1

        if self.finetune_cfg.update_mode == 'local':
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
            assert self.finetune_cfg.update_mode == 'remote'
            self.remote_client = RemoteClient(
                # server_ip='192.168.0.227', 
                server_ip="172.24.68.176",
                server_port=5007,
                exp_folder=self.exp_folder,
            )
            self.replay_buffer = RemoteReplayBuffer(self.remote_client, self.finetune_cfg.buffer_size, num_obs_history=self.num_obs_history, num_privileged_obs_history=self.num_privileged_obs_history, enlarge_when_full=self.finetune_cfg.update_interval * self.finetune_cfg.enlarge_when_full)

        self._make_networks(
            observation_size=self.finetune_cfg.frame_stack * self.obs_size,
            privileged_observation_size=self.finetune_cfg.frame_stack
            * self.privileged_obs_size,
            action_size=self.num_action,
            value_hidden_layer_sizes=self.finetune_cfg.value_hidden_layer_sizes,
            policy_hidden_layer_sizes=self.finetune_cfg.policy_hidden_layer_sizes,
        )
        # self.load_networks('results/stored/toddlerbot_raise_arm_real_world_20250309_172254', data_only=False)

        if self.finetune_cfg.use_residual:
            self._make_residual_policy()
            self._residual_action_scale = self.finetune_cfg.residual_action_scale

        self.obs_history = np.zeros(self.num_obs_history * self.obs_size)
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
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
        self.step_limit = int(5e4)

        self._init_reward()
        self._make_learners()

        self.need_reset = True
        self.learning_stage = "offline"

        self.control_dt = 0.02
        if len(ckpts) > 0:
            self.load_ckpts(ckpts)
        if is_real:
            input("press ENTER to start")

        self.swing_buffer_size = self.finetune_cfg.swing_buffer_size
        self.min_freq, self.max_freq = 0.0, 5.0
        self.fx_buffer = deque(maxlen=self.swing_buffer_size)
        self.fy_buffer = deque(maxlen=self.swing_buffer_size)
        self.fz_buffer = deque(maxlen=self.swing_buffer_size)
        self.pitch_buffer = deque(maxlen=self.swing_buffer_size)
        self.time_buffer = deque(maxlen=self.swing_buffer_size)

        self.t_vals = np.linspace(0, self.swing_buffer_size * self.control_dt,  # Assuming 1/control_dt Hz
                                 self.swing_buffer_size)
        self.target_swing_freq = 0.7 # 0.7Hz for 0.5m pendulum
        self.cycle_time = 1 / self.target_swing_freq

        self.health_xy_force = 10
        self.health_z_force = 40
        self.freq_tolerance = 1.0

        self.phase_signal = np.zeros(2, dtype=np.float32)

    def get_obs(
        self, obs: Obs, command: np.ndarray=None, phase_signal=None, last_action=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        obs_arr = np.concatenate(
            [
                self.phase_signal,  # (2, )
                # command[self.command_obs_indices],  # (3, )
                motor_pos_delta * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel * self.obs_scales.dof_vel,  # (30, )
                self.last_action,  # (6, )
                obs.ang_vel * self.obs_scales.ang_vel,  # (3, )
                obs.euler * self.obs_scales.euler,  # (3, )
            ]
        )
        privileged_obs_arr = np.concatenate(
            [
                self.phase_signal,  # (2, )
                # command[self.command_obs_indices],  # (3, )
                motor_pos_delta * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel * self.obs_scales.dof_vel,  # (30, )
                self.last_action,  # (6, )
                obs.ang_vel * self.obs_scales.ang_vel,  # (3, )
                obs.euler * self.obs_scales.euler,  # (3, )
                obs.ee_force * self.obs_scales.ee_force,  # (3, )
                obs.ee_torque * self.obs_scales.ee_torque,  # (3, )
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

    def _sine_func(self, t, A, freq, phase, offset):
        return A * np.sin(2*np.pi*freq*t + phase) + offset
    
    def _fit_sine_to_buffer(self, buffer):
        """Helper function for sine wave fitting"""
        # import ipdb; ipdb.set_trace()
        if len(buffer) < self.swing_buffer_size // 2:
            return 0, 0, 0, 0, np.inf  # Return defaults for partial buffers
        
        try:
            # Initial parameter guesses
            A_guess = (np.max(buffer) - np.min(buffer))/2
            p0 = [A_guess, self.target_swing_freq, 0, np.mean(buffer)]
            
            bounds = (
                [0, self.target_swing_freq*0.5, -np.pi, -np.inf],
                [A_guess*2, self.target_swing_freq*2, np.pi, np.inf]
            )
            
            params, _ = curve_fit(self._sine_func, 
                                self.t_vals[:len(buffer)], 
                                buffer, 
                                p0=p0, 
                                bounds=bounds,
                                maxfev=1000)
            
            # Calculate RMSE
            fit_vals = self._sine_func(self.t_vals[:len(buffer)], *params)
            rmse = np.sqrt(np.mean((buffer - fit_vals)**2))
            return (*params, rmse)
        except:
            import ipdb; ipdb.set_trace()
            return 0, 0, 0, 0, np.inf

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

        # self.fx_buffer.append(obs.ee_force[0])
        # self.fy_buffer.append(obs.ee_force[1])
        # self.fz_buffer.append(obs.ee_force[2])

        cur_time = time.time()
        if cur_time - self.traj_start_time < self.prep_duration:
            motor_target = np.asarray(
                interpolate_action(cur_time - self.traj_start_time, self.prep_time, self.prep_action)
            )
            self.last_action_target = motor_target[self.action_mask]
            return {}, motor_target, obs

        # if len(self.fx_buffer) < self.swing_buffer_size // 2:
        #     motor_target = self.default_motor_pos.copy()
        #     # Store ee_force data to remote replay buffer
        #     if self.finetune_cfg.update_mode == 'remote':
        #         self.replay_buffer.store(
        #             self.obs_history, 
        #             self.privileged_obs_history, 
        #             self.last_action, 
        #             0, 
        #             False, 
        #             False, 
        #             0, 
        #             raw_obs=deepcopy(obs)
        #         )
        #     return {}, motor_target, obs

        # TODO: verify fit results via plotting
        # if self.finetune_cfg.update_mode == "local":
        #     self.Ax, self.freq_x, self.phase_x, self.offset_x, self.error_x = self._fit_sine_to_buffer(self.fx_buffer)
        # self.Ay, self.freq_y, self.phase_y, self.offset_y, self.error_y = self._fit_sine_to_buffer(self.fy_buffer)
        # self.Az, self.freq_z, self.phase_z, self.offset_z, self.error_z = self._fit_sine_to_buffer(self.fz_buffer)
        
        time_curr = self.step_curr * self.control_dt

        self.phase_signal = self.get_phase_signal(time_curr)
        obs_arr, privileged_obs_arr = self.get_obs(obs)
        if self.total_steps == self.finetune_cfg.offline_total_steps:
            self.switch_learning_stage()
            if len(self.replay_buffer) > 0:
                self.replay_buffer.shift_action(self.action_shift_steps) # TODO: only support continuous data collection, no offline updates in between

        if self.remote_client is not None and self.remote_client.ready_to_update:
            # import ipdb; ipdb.set_trace()
            self.num_updates += 1
            print(f"Updated policy network to {self.num_updates}!")
            assert not torch.allclose(
                self.policy_net.mlp.layers[0].weight,
                self.remote_client.new_state_dict["mlp.layers.0.weight"].to(self.inference_device)
            )
            self.policy_net.load_state_dict(self.remote_client.new_state_dict)
            self.remote_client.ready_to_update = False
            self.need_reset = True

            # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
            motor_target = self.default_motor_pos.copy()
            motor_target[self.action_mask] = self.last_action_target
            return {}, motor_target, obs
        
        if self.learning_stage == "online": # use deterministic action during offline learning
            action_pi, action_real, action_logprob = self.get_action(obs_arr, deterministic=False, is_real=is_real)
        else:
            # data collection stage
            action_pi = self.get_raw_action(obs)
            action_logprob = 0.0

        if self.finetune_cfg.update_mode == 'local':
            reward_dict = self._compute_reward(obs, action_real)
            self.last_last_action = self.last_action.copy()
            self.last_action = action_pi.copy()

            reward = (
                sum(reward_dict.values())
            )  # TODO: verify, why multiply by dt?
            self.logger.log_step(
                reward_dict,
                obs,
                reward=reward,
                action_pi=action_pi.mean(),
                action_real=action_real.mean(),
                fx=obs.ee_force[0],
                fy=obs.ee_force[1],
            )
        else:
            reward = 0.0
        # print(reward_dict)
        time_elapsed = self.timer.elapsed()
        if time_elapsed < self.total_steps * self.control_dt:
            time.sleep(self.total_steps * self.control_dt - time_elapsed)

        if (len(self.replay_buffer) + 1) % 400 == 0:
            print(
                f"Data size: {len(self.replay_buffer)}, Steps: {self.total_steps}, Fps: {self.total_steps / self.timer.elapsed()}"
            )
        time_to_update = (self.learning_stage == "offline" and (len(self.replay_buffer) >= self.finetune_cfg.offline_initial_steps and len(self.replay_buffer) + 1) % self.finetune_cfg.update_interval == 0) or (self.learning_stage == "online" and len(self.replay_buffer) == self.finetune_cfg.online.batch_size - 1)
        truncated = time_to_update and self.finetune_cfg.update_mode == 'local'

        if self.is_truncated():
            truncated = True
            self.current_steps = 0
            print("Truncated! Resetting...")

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
            self.update_policy()
            # import ipdb; ipdb.set_trace()

        if is_real:
            delayed_action = action_real
        else:
            self.action_buffer = np.roll(self.action_buffer, action_real.size)
            self.action_buffer[: action_real.size] = action_real
            delayed_action = self.action_buffer[-self.num_action :]
        # import ipdb; ipdb.set_trace()

        action_target = self.default_action + self.action_scale * delayed_action
        
        if self.filter_type == "ema":
            action_target = exponential_moving_average(
                self.ema_alpha, action_target, self.last_action_target
            )

        # clip diff(action_target, last_action_target) to motor_speed_limits
        action_target = np.clip(
            action_target,
            self.last_action_target - self.motor_speed_limits,
            self.last_action_target + self.motor_speed_limits,
        )
        self.last_action_target = action_target.copy()

        # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
        motor_target = self.default_motor_pos.copy()
        motor_target[self.action_mask] = action_target

        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
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
            self.save_networks()
        save_buffer = input("Save replay buffer? y/n:")
        if save_buffer == "y":
            self.replay_buffer.save_compressed(self.exp_folder)

    def is_truncated(self) -> bool:
        return self.current_steps >= self.step_limit
    
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
        self.last_action = np.zeros(self.num_action)
        self.traj_start_time = time.time()
        self.is_prepared = False
        print("Reset done!")

    def _reward_torso_pitch(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Encourage large torso pitch deviations"""
        return np.clip(np.abs(obs.euler[1]), 0, 0.5)
    
    def _reward_swing_progress(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        return np.abs(obs.euler[1] * obs.ang_vel[1])
    
    def _reward_swing_consistency(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        action_moment = action[self.num_action//2:] - self.last_action[self.num_action//2:]
        return obs.ang_vel[1] * action_moment.mean()
    
    def _reward_action_symmetry(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        return -np.square(action[:self.num_action//2] + action[self.num_action//2:]).mean()
    
    def _reward_swing_spectrum(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward combination of low frequency and large amplitude"""
        self.pitch_buffer.append(obs.euler[1])
        self.time_buffer.append(obs.time)
        
        if len(self.pitch_buffer) < self.swing_buffer_size // 2:  # Minimum window for meaningful FFT
            return 0.0
            
        # Compute FFT
        pitch_series = np.array(self.pitch_buffer)
        dt = np.mean(np.diff(self.time_buffer))
        freqs = np.fft.rfftfreq(len(pitch_series), d=dt)
        fft_vals = np.abs(np.fft.rfft(pitch_series - np.mean(pitch_series)))
        
        # Find dominant frequency
        dominant_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_idx]
        dominant_amp = fft_vals[dominant_idx]
        
        # Frequency reward component (higher reward for lower frequencies)
        freq_reward = np.exp(-5 * (dominant_freq - self.min_freq))
        
        # Amplitude reward component
        amp_reward = np.tanh(dominant_amp / 2.0)  # Scale based on expected amplitudes
        cutoff_idx = np.searchsorted(freqs, self.max_freq)
        high_freq_energy = np.sum(fft_vals[cutoff_idx:])
        # import ipdb; ipdb.set_trace()
        return freq_reward * amp_reward
    
    def _reward_fx_sine_amplitude(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for large Fx amplitude"""
        return np.clip(self.Ax, 0, 10)  # Clip to prevent exploding rewards

    def _reward_fx_sine_fit(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for good Fx sine fit"""
        return 1 / (1 + self.error_x)  # Inverse of RMSE

    def _reward_fz_sine_amplitude(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for appropriate Fz amplitude"""
        # return np.exp(-0.5*(self.Az - self.Az_target)**2)  # Gaussian around target amplitude
        return np.clip(self.Az, 0, 10)  # Clip to prevent exploding rewards

    def _reward_fz_sine_fit(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward for good Fz sine fit"""
        return 1 / (1 + self.error_z)

    def _reward_fy_suppression(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Penalize Fy deviations"""
        self.fy_buffer.append(obs.ee_force[1])
        fy_amp = np.max(self.fy_buffer) - np.min(self.fy_buffer)
        # return -np.tanh(10*(fy_amp - self.fy_tolerance))
        return -np.tanh(10*(fy_amp))

    def _reward_phase_alignment(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward proper Fx-Fz phase relationship"""

        phase_diff = np.abs(self.phase_x - self.phase_z) % (2*np.pi)
        return np.cos(np.minimum(phase_diff, 2*np.pi-phase_diff))

    def _reward_frequency_consistency(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Penalize frequency deviations from target"""
        freq_error = np.abs(self.freq_x - self.target_swing_freq)
        return -freq_error / self.freq_tolerance

    def _reward_energy_efficiency(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Penalize high-frequency components in Fx"""
        if len(self.fx_buffer) < 50:
            return 0
            
        fft_vals = np.abs(rfft(self.fx_buffer))
        high_freq_energy = np.sum(fft_vals[5:])  # Above 5th harmonic
        return -high_freq_energy / 1000

    def _reward_swing_symmetry(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        """Reward symmetric positive/negative Fx swings"""
        self.fx_buffer.append(obs.ee_force[0])
        pos_peak = np.max(self.fx_buffer)
        neg_peak = np.abs(np.min(self.fx_buffer))
        return 1 - np.abs(pos_peak - neg_peak)/(pos_peak + neg_peak + 1e-6)

    def _reward_arm_action_rate(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - self.last_action)
        reward = -np.mean(error)
        return reward

    def _reward_arm_action_acc(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - 2 * self.last_action + self.last_last_action)
        reward = -np.mean(error)
        return reward