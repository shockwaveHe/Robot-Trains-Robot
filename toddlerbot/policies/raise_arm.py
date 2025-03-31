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
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer, RemoteReplayBuffer
from toddlerbot.finetuning.server_client import RemoteClient
from toddlerbot.finetuning.utils import Timer
from toddlerbot.utils.math_utils import (
    exponential_moving_average,
    inverse_exponential_moving_average,
)
from toddlerbot.finetuning.logger import FinetuneLogger


class RaiseArmPolicy(MJXFinetunePolicy, policy_name="raise_arm"):
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
    ):
        if env_cfg is None:
            env_cfg = get_env_config("raise_arm", exp_folder)
        if finetune_cfg is None:
            finetune_cfg = get_finetune_config("raise_arm", exp_folder)
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
            need_warmup=False,
        )
        # self.control_dt = 0.1
        self.robot = robot
        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.finetune_cfg.update_mode == "local"
            else "cpu"
        )
        self.inference_device = (
            "cuda"
            if torch.cuda.is_available() and self.finetune_cfg.update_mode == "local"
            else "cpu"
        )
        self.rng = np.random.default_rng()
        self.num_obs_history = self.cfg.obs.frame_stack
        self.obs_size = self.finetune_cfg.num_single_obs

        self.is_real = False  # hardcode to False for zmq, etc.
        self.is_paused = False
        self.active_motor_idx = [16, 19, 21, 23, 26, 28]
        self.action_delta_limit = 2 * self.control_dt
        self.hand_z_dist_base = 0.266
        self.arm_radius = 0.1685
        self.hand_z_dist_terminal = 0.5
        self.action_mask = np.array(self.active_motor_idx)
        self.num_action = self.action_mask.shape[0]

        self.default_motor_pos[18] = np.pi / 2
        self.default_motor_pos[20] = -np.pi / 2
        self.default_motor_pos[25] = -np.pi / 2
        self.default_motor_pos[27] = np.pi / 2

        self.default_action = self.default_motor_pos[self.action_mask]
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
        # self.load_networks('results/stored/toddlerbot_raise_arm_real_world_20250309_172254', data_only=False)

        if self.finetune_cfg.use_residual:
            self._make_residual_policy()
            self._residual_action_scale = self.finetune_cfg.residual_action_scale

        self.obs_history = np.zeros(
            self.num_obs_history * self.obs_size, dtype=np.float32
        )
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size, dtype=np.float32
        )

        self.logger = FinetuneLogger(self.exp_folder)

        self.num_updates = 0
        self.total_steps, self.current_steps = 0, 0
        self.step_limit = int(5e4)

        self._init_reward()
        self._make_learners()

        self.need_reset = True
        self.learning_stage = "offline"

        self.max_hand_z = -np.inf
        self.sim = MuJoCoSim(robot, vis_type=self.finetune_cfg.sim_vis_type, n_frames=1)

        if len(ckpts) > 0:
            self.load_ckpts(ckpts)
        input("press ENTER to start")

        self.phase_signal = np.zeros(2, dtype=np.float32)

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

    # TODO: add a timeout and truncation?
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

        cur_time = time.time()
        if cur_time - self.traj_start_time < self.prep_duration:
            motor_target = np.asarray(
                interpolate_action(
                    cur_time - self.traj_start_time, self.prep_time, self.prep_action
                )
            )
            self.last_action_target = motor_target[self.action_mask]
            return {}, motor_target, obs

        if self.finetune_cfg.update_mode == "local":
            self.sim.set_motor_angles(obs.motor_pos)
            self.sim.forward()
            hand_pos = self.sim.get_hand_pos()
            hand_z_dist = np.array([hand_pos["left"][2], hand_pos["right"][2]])
            if hand_z_dist.max() > self.max_hand_z:
                self.max_hand_z = hand_z_dist.max()
                print(f"max z reached: {self.max_hand_z}")
            obs.hand_z_dist = hand_z_dist

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
                obs_arr, deterministic=False, is_real=is_real
            )
        else:
            # data collection stage
            action_pi = self.get_raw_action(obs)
            action_logprob = 0.0

        if self.finetune_cfg.update_mode == "local":
            reward_dict = self._compute_reward(obs, action_real)
            self.last_last_action = self.last_action.copy()
            self.last_action = action_pi.copy()

            reward = sum(reward_dict.values())  # TODO: verify, why multiply by dt?
            self.logger.log_step(
                reward_dict,
                obs,
                reward=reward,
                hand_z_dist_left=obs.hand_z_dist[0],
                hand_z_dist_right=obs.hand_z_dist[1],
                action_pi=action_pi.mean(),
                action_real=action_real.mean(),
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
        time_to_update = (
            self.learning_stage == "offline"
            and (
                len(self.replay_buffer) >= self.finetune_cfg.offline_initial_steps
                and len(self.replay_buffer) + 1
            )
            % self.finetune_cfg.update_interval
            == 0
        ) or (
            self.learning_stage == "online"
            and len(self.replay_buffer) == self.finetune_cfg.online.batch_size - 1
        )
        truncated = time_to_update and self.finetune_cfg.update_mode == "local"

        if self.is_truncated():
            truncated = True
            self.current_steps = 0
            print("Truncated! Resetting...")

        self.replay_buffer.store(
            obs_arr,
            privileged_obs_arr,
            action_pi,
            reward,
            self.is_done(obs),
            truncated,
            action_logprob,
            raw_obs=deepcopy(obs),
        )

        if truncated:
            self.update_policy()
            self.need_reset = True
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
        self.total_steps += 1
        self.current_steps += 1
        self.step_curr += 1
        self.timer.start()
        control_inputs = {}
        self.control_inputs = control_inputs

        return control_inputs, motor_target, obs

    def close(self):
        self.sim.close()
        self.logger.close()
        save_networks = input("Save networks? y/n:")
        while save_networks not in ["y", "n"]:
            save_networks = input("Save networks? y/n:")
        if save_networks == "y":
            self.save_networks()
        save_buffer = input("Save replay buffer? y/n:")
        if save_buffer == "y":
            self.replay_buffer.save_compressed(self.exp_folder)

    def is_done(self, obs: Obs) -> bool:
        # TODO potentially some bugs after is done
        return obs.hand_z_dist.max() > self.hand_z_dist_terminal

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
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_last_action = np.zeros(self.num_action, dtype=np.float32)
        self.last_action_target = self.default_action
        self.last_raw_action = None
        self.traj_start_time = time.time()
        self.is_prepared = False
        print("Reset done!")

    def _reward_survival(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # if self.is_done(obs):
        #     print("done reward")
        # return self.is_done(obs)
        return 1.0

    # h = r * (1 - cos(theta))
    # theta = acos(1 - h / r)
    def _reward_arm_position(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        return np.arccos(
            1
            - np.clip(obs.hand_z_dist - self.hand_z_dist_base, 0, np.inf)
            / self.arm_radius
        ).sum()

    def _reward_action_rate(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - self.last_action)
        reward = -np.mean(error)
        return reward

    def _reward_action_acc(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - 2 * self.last_action + self.last_last_action)
        reward = -np.mean(error)
        return reward
