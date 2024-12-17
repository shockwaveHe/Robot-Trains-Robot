import math
import os
import numpy as np
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from toddlerbot.sim import Obs
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action
from toddlerbot.finetuning.networks import make_ppo_networks, load_jax_params, load_jax_params_into_pytorch
from scipy.spatial.transform import Rotation
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.utils.math_utils import euler2quat
from pathlib import Path
from numba import njit
from toddlerbot.utils.comm_utils import ZMQNode

class MJXFinetunePolicy(MJXPolicy, policy_name="finetune"):
    def __init__(self, name, robot: Robot, *args, **kwargs):
        super().__init__(name, robot, *args, **kwargs)
        self.robot = robot
        self.ppo_networks = None
        self.replay_buffer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "seed" in kwargs:
            self.rng = np.random.default_rng(kwargs["seed"])
        else:
            self.rng = np.random.default_rng()
        self.motion_ref = WalkZMPReference(
            robot,
            self.cfg.sim.timestep * self.cfg.action.n_frames,
            self.cfg.action.cycle_time,
            self.cfg.action.waist_roll_max,
        )
        self.command_range = np.array(self.cfg.commands.command_range) # TODO: set command range to x>0, y~0 and z=0, turn_change = 0
        self.deadzone = (
            np.array(self.cfg.commands.deadzone)
            if len(self.cfg.commands.deadzone) > 1
            else self.cfg.commands.deadzone[0]
        )
        self.zero_chance = self.cfg.commands.zero_chance
        self.turn_chance = self.cfg.commands.turn_chance
        self.train_cfg = PPOConfig()
        self.last_action = np.zeros(self.num_action)
        self.last_last_action = np.zeros(self.num_action)
        self.last_obs = None
        self.last_privileged_obs = None
        self.last_reward = 0.0

        self.ref_start_idx = 13
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs

        print(f"Observation size: {self.obs_size}, Privileged observation size: {self.privileged_obs_size}")
        self.replay_buffer = OnlineReplayBuffer(self.device, self.obs_size * self.num_obs_history, self.privileged_obs_size * self.num_privileged_obs_history, self.num_action, self.cfg.finetune.buffer_size) # TODO: add priviledged obs to buffer

        print(f"Buffer size: {self.cfg.finetune.buffer_size}")
        self.ppo_networks = make_ppo_networks(
            observation_size=self.cfg.obs.frame_stack * self.obs_size,
            privileged_observation_size=self.cfg.obs.frame_stack * self.privileged_obs_size,
            action_size=self.num_action,
            value_hidden_layer_sizes=self.train_cfg.value_hidden_layer_sizes,
            policy_hidden_layer_sizes=self.train_cfg.policy_hidden_layer_sizes,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if len(self.ckpt) > 0:
            run_name = f"{self.robot.name}_{self.name}_ppo_{self.ckpt}"
            policy_path = os.path.join("results", run_name, "best_policy")
            if not os.path.exists(policy_path):
                policy_path = os.path.join("results", run_name, "policy")
        else:
            policy_path = os.path.join(
                "toddlerbot",
                "policies",
                "checkpoints",
                f"{self.robot.name}_walk_policy",
            )
        print(f"Loading pretrained model from {policy_path}")
        jax_params = load_jax_params(policy_path)
        load_jax_params_into_pytorch(self.ppo_networks.policy_network, jax_params[1]["params"])

        self.obs_history = np.zeros(self.num_obs_history * self.obs_size)
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        self.zmq_receiver = ZMQNode(type="receiver")
        self._init_reward()

    def _sample_command(
        self, last_command: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Randomly sample an index from the command list
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self.rng.uniform(
                low=self.command_range[:5, 0],
                high=self.command_range[:5, 1],
                size=(5,)
            )
            pose_command[:5] = 0.0  # TODO: Bring the random pose sampling back

        def sample_walk_command():
            # Sample random angles uniformly between 0 and 2*pi
            theta = self.rng.uniform(low=0, high=2 * np.pi, size=(1,))
            # Parametric equation of ellipse
            x_max = np.where(
                np.sin(theta) > 0, self.command_range[5][1], -self.command_range[5][0]
            )
            x = self.rng.uniform(low=self.deadzone, high=x_max, size=(1,)
            ) * np.sin(theta)
            y_max = np.where(
                np.cos(theta) > 0, self.command_range[6][1], -self.command_range[6][0]
            )
            y = self.rng.uniform(
                low=self.deadzone, high=y_max, size=(1,)
            ) * np.cos(theta)
            z = np.zeros(1)
            return np.concatenate([x, y, z])
        def sample_turn_command():
            x = np.zeros(1)
            y = np.zeros(1)
            z = np.where(
                self.rng.uniform((1,)) < 0.5,
                self.rng.uniform(
                    low=self.deadzone,
                    high=self.command_range[7][1],
                    size=(1,),
                ),
                -self.rng.uniform(
                    low=self.deadzone,
                    high=-self.command_range[7][0],
                    size=(1,),
                ),
            )
            return np.concatenate([x, y, z])

        random_number = self.rng.uniform((1,))
        walk_command = np.where(
            random_number < self.zero_chance,
            np.zeros(3),
            np.where(
                random_number < self.zero_chance + self.turn_chance,
                sample_turn_command(),
                sample_walk_command(),
            ),
        )
        command = np.concatenate([pose_command, walk_command])
        return command
    
    def get_obs(self, obs: Obs) -> Tuple[np.ndarray, np.ndarray]:

        command = self.fixed_command
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        self.state_ref = np.asarray(self.motion_ref.get_state_ref(self.state_ref, 0.0, command))
        motor_pos_error = obs.motor_pos - self.state_ref[self.ref_start_idx : self.ref_start_idx + self.robot.nu]
        
        obs_arr = np.concatenate(
            [
                self.phase_signal,
                command[self.command_obs_indices],
                motor_pos_delta * self.obs_scales.dof_pos,
                obs.motor_vel * self.obs_scales.dof_vel,
                self.last_action,
                # motor_pos_error,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
            ]
        )
        privileged_obs_arr = np.concatenate(
            [
                self.phase_signal,
                command[self.command_obs_indices],
                motor_pos_delta * self.obs_scales.dof_pos,
                obs.motor_vel * self.obs_scales.dof_vel,
                self.last_action, # TODO: set last action
                motor_pos_error,
                obs.lin_vel * self.obs_scales.lin_vel, 
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
                obs.ee_force * self.obs_scales.ee_force,
                obs.ee_torque * self.obs_scales.ee_torque,
                # self.state_ref[-2:], # TODO: add stance back?
                # TODO: push lin and ang vel
            ]
        )
        # TODO: verify correct
        obs = np.roll(self.obs_history, obs_arr.size)
        obs[:obs_arr.size] = obs_arr
        privileged_obs = np.roll(self.privileged_obs_history, privileged_obs_arr.size)
        privileged_obs[:privileged_obs_arr.size] = privileged_obs_arr

        return obs, privileged_obs
    
    def get_action(self, obs_arr: np.ndarray, deterministic: bool = True, is_real: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.ppo_networks.policy_network(obs_tensor)
            if deterministic:
                # Deterministic: use mode
                actions = self.ppo_networks.parametric_action_distribution.mode(logits).cpu().numpy().flatten()
                return actions, {}
            else:
                # Stochastic: sample raw pre-tanh actions
                raw_actions = self.ppo_networks.parametric_action_distribution.sample_no_postprocessing(logits)
                log_prob = self.ppo_networks.parametric_action_distribution.log_prob(logits, raw_actions)
                postprocessed_actions = self.ppo_networks.parametric_action_distribution.postprocess(raw_actions)
                return postprocessed_actions, {
                    'log_prob': log_prob,
                    'raw_action': raw_actions
                }

    def reset(self, obs:Obs = None):
        # mjx policy reset
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.is_standing = True
        self.command_list = []
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.step_curr = 0
        if obs is not None:
            # TODO: more things to reset?
            path_pos = np.zeros(3)
            # path_yaw = self.rng.uniform(low=0, high=2 * np.pi, size=(1,))
            path_yaw = obs.euler[2] # TODO: only change in a small range
            path_euler = np.array([0.0, 0.0, np.degrees(path_yaw)])
            path_quat = euler2quat(path_euler) # TODO: verify usage, wxyz or xyzw?
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
            # motor_pos = obs.joint_pos[self.q_start_idx + self.motor_indices]
            # joint_pos = obs.joint_pos[self.q_start_idx + self.joint_indices]
            motor_pos = np.zeros_like(self.default_motor_pos)
            joint_pos = np.zeros_like(self.default_joint_pos)
            stance_mask = np.ones(2)

            state_ref = np.concatenate(
                [path_pos, path_quat, lin_vel, ang_vel, motor_pos, joint_pos, stance_mask]
            )
            self.fixed_command = self._sample_command()
            self.state_ref = np.asarray(self.motion_ref.get_state_ref(state_ref, 0.0, self.fixed_command))

    def is_done(self, obs: Obs) -> bool:
        pass # is done is handled in sim
    
    def step(self, obs:Obs, is_real:bool = False):

        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 0.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        time_curr = self.step_curr * self.control_dt

        msg = self.zmq_receiver.get_msg()
        control_inputs: Dict[str, float] = {}
        if len(self.control_inputs) > 0:
            control_inputs = self.control_inputs
        elif msg is not None and msg.control_inputs is not None:
            print(f'obs ee force: {obs.ee_force}')
            control_inputs = msg.control_inputs
            obs.ee_force = msg.arm_force
            obs.ee_torque = msg.arm_torque
            obs.lin_vel = msg.lin_vel

        if len(control_inputs) == 0:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)


        self.phase_signal = self.get_phase_signal(time_curr)
        obs_arr, privileged_obs_arr = self.get_obs(obs)
        reward_dict = self._compute_reward(obs, self.last_action)
        reward = sum(reward_dict.values()) * self.control_dt # TODO: verify
        action, _ = self.get_action(obs_arr, deterministic=True, is_real=is_real)
        
        self.replay_buffer.store(self.last_obs, self.last_privileged_obs, self.last_action, self.last_reward, obs_arr, privileged_obs_arr, action, self.is_done(obs))
        self.last_reward = reward
        self.last_obs = obs_arr.copy()
        self.last_privileged_obs = privileged_obs_arr.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        if is_real:
            delayed_action = action
        else:
            self.action_buffer = np.roll(self.action_buffer, action.size)
            self.action_buffer[: action.size] = action
            delayed_action = self.action_buffer[-self.num_action :]

        action_target = self.default_action + self.action_scale * delayed_action
        self.last_action_target = action_target.copy()

        # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
        motor_target = self.default_motor_pos.copy()
        motor_target[self.action_mask] = action_target

        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.command_list.append(command)
        self.last_action = delayed_action
        self.step_curr += 1

        return control_inputs, motor_target
    
    def _init_reward(self) -> None:
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        reward_scale_dict = asdict(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions: List[Callable[..., np.ndarray]] = []
        self.reward_scales = np.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            if getattr(self, "_reward_" + name, None) is None:
                self.reward_names.remove(name)
                print(f"Warning: reward function _reward_{name} not found")
                continue
            self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales[i] = scale

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.tracking_sigma = self.cfg.rewards.tracking_sigma

    def _compute_reward(
        self, obs: Obs, action: np.ndarray
    ):

        reward_dict: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = self.reward_functions[i](obs, action)

        return reward_dict
    
    # @njit
    def _reward_torso_pos(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_pos = obs.pos[:2] # TODO: no torso pos
        torso_pos_ref = self.motion_ref[:2]
        error = np.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = np.exp(-200.0 * error**2) # TODO: scale
        return reward
    
    def _reward_torso_quat(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_euler = obs.euler
        torso_rot = Rotation.from_euler("zxy", torso_euler)
        torso_quat = torso_rot.as_quat()
        path_quat_ref = self.state_ref[3:7]
        path_rot = Rotation.from_quat(path_quat_ref)
        
        waist_joint_pos = self.state_ref[self.ref_start_idx + self.robot.nu + self.waist_motor_indices]
        waist_euler = np.array([waist_joint_pos[0], 0.0, waist_joint_pos[1]])
        waist_rot = Rotation.from_euler("xyz", waist_euler)
        # torso_quat_ref = math.quat_mul(
        #     path_quat_ref, math.quat_inv(waist_quat)
        # )
        torso_rot = path_rot * waist_rot.inv()
        torso_quat_ref = torso_rot.as_quat()

        # Quaternion dot product (cosine of the half-angle)
        dot_product = np.sum(torso_quat * torso_quat_ref, axis=-1)
        # Ensure the dot product is within the valid range
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Quaternion angle difference
        angle_diff = 2.0 * np.arccos(np.abs(dot_product))
        reward = np.exp(-20.0 * (angle_diff**2))
        return reward

    def _reward_lin_vel_xy(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[:2] # TODO: rotate to local? or get it from treadmill
        # TODO: change treadmill speed according to force x, or estimate from IMU + joint_position
        # TODO: compare which is better
        lin_vel_ref = self.state_ref[7:9]
        error = np.linalg.norm(lin_vel - lin_vel_ref, axis=-1)
        reward = np.exp(-self.tracking_sigma * error**2)
        return reward
    
    def _reward_lin_vel_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[2] # TODO: change to normal force
        lin_vel_ref = self.state_ref[9]
        error = np.abs(lin_vel - lin_vel_ref)
        reward = np.exp(-self.tracking_sigma * error**2)
        return reward
    
    def _reward_ang_vel_xy(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ang_vel = obs.ang_vel[:2]
        ang_vel_ref = self.state_ref[10:12]
        error = np.linalg.norm(ang_vel - ang_vel_ref, axis=-1)
        reward = np.exp(-self.tracking_sigma / 4 * error**2)
        return reward
    
    def _reward_ang_vel_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ang_vel = obs.ang_vel[2]
        ang_vel_ref = self.state_ref[12]
        error = np.abs(ang_vel - ang_vel_ref)
        reward = np.exp(-self.tracking_sigma / 4 * error**2)
        return reward
    
    def _reward_leg_motor_pos(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        motor_pos = obs.motor_pos[self.leg_motor_indices]
        motor_pos_ref = self.state_ref[self.ref_start_idx + self.leg_motor_indices]
        error = motor_pos - motor_pos_ref
        reward = -np.mean(error**2) # TODO: why not exp?
        return reward
    
    # def _reward_motor_torque(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: how to get motor torque?
    # def _reward_energy(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: how to get energy?

    def _reward_leg_action_rate(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - self.last_action)
        reward = -np.mean(error)
        return reward
    
    # def _reward_leg_action_acc(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: store last last action?

    # def _reward_feet_contact(self, obs: Obs, action: np.ndarray) -> np.ndarray:
    # def _reward_collision(self, obs: Obs, action: np.ndarray) -> np.ndarray:
    def _reward_survival(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        is_done = self.is_done(obs)
        reward = -np.where(is_done, 1.0, 0.0)
        return reward