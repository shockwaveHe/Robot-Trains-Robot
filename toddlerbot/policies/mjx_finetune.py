import os
import time
import numpy as np
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from copy import deepcopy
import torch
import pickle
import numpy.typing as npt
from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.dynamics import DynamicsNetwork, BaseDynamics
from toddlerbot.finetuning.utils import Timer
from toddlerbot.motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action
from toddlerbot.finetuning.networks import load_jax_params, load_jax_params_into_pytorch
import toddlerbot.finetuning.networks as networks
from toddlerbot.finetuning.abppo import AdaptiveBehaviorProximalPolicyOptimization, ABPPO_Offline_Learner
from scipy.spatial.transform import Rotation
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.utils.math_utils import euler2quat, euler2mat, mat2euler
from toddlerbot.utils.comm_utils import ZMQNode, ZMQMessage
from toddlerbot.finetuning.logger import FinetuneLogger
from pyvicon_datastream import tools

class MJXFinetunePolicy(MJXPolicy, policy_name="finetune"):
    def __init__(self, name, robot: Robot, init_motor_pos: npt.NDArray[np.float32], ckpt: str, ip: str, joystick, fixed_command, env_cfg, finetune_cfg: FinetuneConfig, *args, **kwargs):
        # set these before super init
        self.is_stopped = False
        self.finetune_cfg: FinetuneConfig = finetune_cfg


        self.num_privileged_obs_history = self.finetune_cfg.frame_stack
        self.privileged_obs_size = self.finetune_cfg.num_single_privileged_obs
        self.privileged_obs_history_size = self.privileged_obs_size * self.num_privileged_obs_history
        self.replay_buffer = None
        super().__init__(name, robot, init_motor_pos, "", joystick, fixed_command, env_cfg, *args, **kwargs)
        self.robot = robot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_device = "cpu"
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

        self.ref_start_idx = 13

        self.num_obs_history = self.cfg.obs.frame_stack
        self.obs_size = self.finetune_cfg.num_single_obs
        self.last_action = np.zeros(self.num_action)
        self.last_last_action = np.zeros(self.num_action)

        print(f"Observation size: {self.obs_size}, Privileged observation size: {self.privileged_obs_size}")
        self.replay_buffer = OnlineReplayBuffer(self.device, self.obs_size * self.num_obs_history, self.privileged_obs_size * self.num_privileged_obs_history, self.num_action, self.finetune_cfg.buffer_size) # TODO: add priviledged obs to buffer
        # import pickle
        # with open('buffer_mock.pkl', 'rb') as f:
        #     self.replay_buffer: OnlineReplayBuffer = pickle.load(f)
        print(f"Buffer size: {self.finetune_cfg.buffer_size}")
        self._make_networks(
            observation_size=self.finetune_cfg.frame_stack * self.obs_size,
            privileged_observation_size=self.finetune_cfg.frame_stack * self.privileged_obs_size,
            action_size=self.num_action,
            value_hidden_layer_sizes=self.finetune_cfg.value_hidden_layer_sizes,
            policy_hidden_layer_sizes=self.finetune_cfg.policy_hidden_layer_sizes,
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
        load_jax_params_into_pytorch(self.policy_net, jax_params[1]["params"])

        self.obs_history = np.zeros(self.num_obs_history * self.obs_size)
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        self.zmq_receiver = ZMQNode(type="receiver")
        assert len(ip) > 0, "Please provide the IP address of the sender"
        self.zmq_sender = ZMQNode(type="sender", ip=ip)
        self.logger = FinetuneLogger(self.exp_folder)

        self.total_steps = 0
        self.timer = Timer()
        self._init_reward()
        self._init_tracker()

        if len(ckpt) > 0:
            self.load_networks(ckpt)

        self._make_learners()

        self.sim = MuJoCoSim(robot, vis_type=self.finetune_cfg.sim_vis_type, hang_force=0.0)
        input('press ENTER to start')

    def close(self):
        self.sim.close()
        self.logger.close()
        self.zmq_receiver.close()
        save_networks= input("Save networks? y/n:")
        while save_networks not in ['y', 'n']:
            save_networks = input("Save networks? y/n:")
        if save_networks == 'y':
            self.save_networks()

    def _make_networks(
        self,
        observation_size: int,
        privileged_observation_size: int,
        action_size: int,
        preprocess_observations_fn: Callable = lambda x, y: x,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation_fn: Callable = torch.nn.SiLU,  # PyTorch equivalent of linen.swish is SiLU
    ):
        """Make PPO networks with a PyTorch implementation."""

        # Create policy network
        # Note: Ensure that your policy MLP ends with parametric_action_distribution.param_size units
        self.policy_net = networks.GaussianPolicyNetwork(
            observation_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=policy_hidden_layer_sizes,
            action_size=action_size,
            activation_fn=activation_fn
        ).to(self.inference_device)
        self.policy_net_opt = torch.compile(self.policy_net)
        # Create value network
        self.value_net = networks.ValueNetwork(
            observation_size=privileged_observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn
        ).to(self.device)

        Q_net_cls = networks.DoubleQNetwork if self.finetune_cfg.use_double_q else networks.QNetwork
        self.Q_net = Q_net_cls(
            observation_size=privileged_observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn
        ).to(self.device)

        self.dynamics_net = DynamicsNetwork(
            observation_size=privileged_observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn
        ).to(self.device)

        self.dynamics = BaseDynamics(self.device, self.dynamics_net, self.finetune_cfg)

    def _init_tracker(self):
        self.tracker = tools.ObjectTracker(self.finetune_cfg.vicon_ip)
        self.mocap_marker_offset = self.finetune_cfg.mocap_marker_offset
        self.tracking_alpha = self.finetune_cfg.tracking_alpha
        self.tracking_tf_matrix = self.finetune_cfg.tracking_tf_matrix

        for _ in range(100):
            result = self.tracker.get_position(self.finetune_cfg.object_name)
            prev_euler = np.array(result[2][0][5:8])

        self.R_default = euler2mat(prev_euler)
        self.init_euler = prev_euler.copy()
        self.prev_unwrapped = prev_euler.copy()
        self.prev_lin_vel = np.zeros(3)
        self.prev_ang_vel = np.zeros(3)

    def get_tracking_data(self):
        result = self.tracker.get_position(self.finetune_cfg.object_name)
        current_time = time.time()
        current_pos = np.array(result[2][0][2:5]) / 1000
        current_euler = np.array(result[2][0][5:8])
        # current_euler = np.array([current_euler[2], current_euler[1], current_euler[0]])
        
        # Calculate rotation and offset
        R_current = euler2mat(current_euler)
        current_euler = mat2euler(R_current @ self.R_default.T)
        offset_current = R_current @ self.R_default.T @ self.mocap_marker_offset
        current_pos -= offset_current
        
        # Initialize angular velocity with zeros
        ang_vel = np.zeros(3)
        
        # Calculate time difference
        if hasattr(self, 'prev_time'):
            dt = current_time - self.prev_time
        else:
            self.prev_time = current_time
            return np.zeros(3), np.zeros(3), np.zeros(3), current_time
        
        # Calculate linear velocities (x, y, z)
        lin_vel = (current_pos - self.prev_pos) / dt if hasattr(self, 'prev_pos') else np.zeros(3)
        
        # Calculate angular velocities
        if hasattr(self, 'prev_euler'):
            # Compute Euler angle derivatives
            delta_euler = current_euler - self.prev_euler
            dphi_dt, dtheta_dt, dpsi_dt = delta_euler / dt

            # Get current Euler angles
            _, beta, gamma = current_euler

            # Compute trigonometric values
            sin_beta = np.sin(beta)
            cos_beta = np.cos(beta)
            sin_gamma = np.sin(gamma)
            cos_gamma = np.cos(gamma)

            # Construct transformation matrix
            E = np.array([
                [cos_beta * cos_gamma, -sin_gamma, 0],
                [cos_beta * sin_gamma, cos_gamma, 0],
                [-sin_beta, 0, 1]
            ])
            # Calculate angular velocity
            ang_vel = E @ np.array([dphi_dt, dtheta_dt, dpsi_dt])
        else:
            ang_vel = np.zeros(3)

        # Update previous values
        self.prev_time = current_time
        self.prev_pos = current_pos
        self.prev_euler = current_euler.copy()

        # EMA filtering
        lin_vel = self.tracking_alpha * lin_vel + (1 - self.tracking_alpha) * self.prev_lin_vel
        ang_vel = self.tracking_alpha * ang_vel + (1 - self.tracking_alpha) * self.prev_ang_vel
        self.prev_lin_vel = lin_vel
        self.prev_ang_vel = ang_vel
        
        # Transform velocities
        if self.tracking_tf_matrix is not None:
            lin_vel = self.tracking_tf_matrix @ lin_vel
            ang_vel = self.tracking_tf_matrix @ ang_vel
            current_euler = self.tracking_tf_matrix @ current_euler

        return lin_vel, ang_vel, current_euler, current_time
    
    def save_networks(self):
        policy_path = os.path.join(self.exp_folder, "policy")
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        torch.save(self.policy_net.state_dict(), os.path.join(policy_path, "policy_net.pth"))
        torch.save(self.value_net.state_dict(), os.path.join(policy_path, "value_net.pth"))
        torch.save(self.Q_net.state_dict(), os.path.join(policy_path, "Q_net.pth"))
        torch.save(self.dynamics_net.state_dict(), os.path.join(policy_path, "dynamics_net.pth"))
        self.logger.save_state(os.path.join(self.exp_folder, "logger.pkl"))
        save_buffer = input("Save replay buffer? y/n:")
        if save_buffer == 'y':
            with open(os.path.join(self.exp_folder, "buffer.pkl"), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
    
    def load_networks(self, policy_path):
        org_policy_net = deepcopy(self.policy_net)
        policy_path = os.path.join(policy_path, "policy")
        assert os.path.exists(policy_path), f"Path {policy_path} does not exist"
        self.policy_net.load_state_dict(torch.load(os.path.join(policy_path, "policy_net.pth")))
        self.value_net.load_state_dict(torch.load(os.path.join(policy_path, "value_net.pth")))
        self.Q_net.load_state_dict(torch.load(os.path.join(policy_path, "Q_net.pth")))
        self.dynamics_net.load_state_dict(torch.load(os.path.join(policy_path, "dynamics_net.pth")))
        assert not torch.allclose(org_policy_net.mlp.layers[0].weight, self.policy_net.mlp.layers[0].weight), "Policy network not loaded correctly"
        self.logger.load_state(os.path.join(self.exp_folder, "logger.pkl"))
        print(f"Loaded pretrained model from {policy_path}")
        if os.path.exists(os.path.join(self.exp_folder, "buffer.pkl")):
            with open(os.path.join(self.exp_folder, "buffer.pkl"), 'rb') as f:
                self.replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer from {self.exp_folder}")

    def _make_learners(self):
        """Make PPO learners with a PyTorch implementation."""
        self.abppo = AdaptiveBehaviorProximalPolicyOptimization(self.device, self.policy_net, self.finetune_cfg)
        self.abppo_offline_learner = ABPPO_Offline_Learner(self.device, self.finetune_cfg, self.abppo, self.Q_net, self.value_net, self.dynamics, self.logger)

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
    
    def get_obs(self, obs: Obs, command: np.ndarray, phase_signal = None, last_action = None) -> Tuple[np.ndarray, np.ndarray]:
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        # state_ref_ds = np.asarray(self.motion_ref.get_state_ref_ds(self.state_ref, 0.0, command))
        self.state_ref = np.asarray(self.motion_ref.get_state_ref(self.state_ref, 0.0, command))
        
        obs_arr = np.concatenate(
            [
                self.phase_signal if phase_signal is None else phase_signal, # (2, )
                command[self.command_obs_indices], # (3, )
                motor_pos_delta * self.obs_scales.dof_pos, # (30, )
                obs.motor_vel * self.obs_scales.dof_vel, # (30, )
                self.last_action if last_action is None else last_action, # (12, )
                # motor_pos_error,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel, # (3, )
                obs.euler * self.obs_scales.euler, # (3, )
            ]
        )
        privileged_obs_arr = np.concatenate(
            [
                self.phase_signal if phase_signal is None else phase_signal, # (2, )
                command[self.command_obs_indices], # (3, )
                motor_pos_delta * self.obs_scales.dof_pos, # (30, )
                obs.motor_vel * self.obs_scales.dof_vel, # (30, )
                self.last_action if last_action is None else last_action, # (12, )
                obs.motor_pos, # (30, ) change from motor_pos_error
                obs.lin_vel * self.obs_scales.lin_vel, # (3, )
                obs.ang_vel * self.obs_scales.ang_vel, # (3, )
                obs.euler * self.obs_scales.euler, # (3, )
                obs.ee_force * self.obs_scales.ee_force, # (3, )
                obs.ee_torque * self.obs_scales.ee_torque, # (3, )
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)
        self.obs_history[:obs_arr.size] = obs_arr
        self.privileged_obs_history = np.roll(self.privileged_obs_history, privileged_obs_arr.size)
        self.privileged_obs_history[:privileged_obs_arr.size] = privileged_obs_arr

        return self.obs_history, self.privileged_obs_history
    
    def get_action(self, obs_arr: np.ndarray, deterministic: bool = True, is_real: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        # import ipdb; ipdb.set_trace()
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32).to(self.inference_device).squeeze(0)
        with torch.no_grad():
            action_dist = self.policy_net_opt(obs_tensor) # use the compiled model for faster inference, the parameters are shared with the original model, see test_compile.py
            if deterministic:
                # Deterministic: use mode
                actions = action_dist.base_dist.mode
                for transform in action_dist.transforms:
                    actions = transform(actions)
                actions = actions.cpu().numpy().flatten()

                return actions, {}
            else:
                # Stochastic: sample raw pre-tanh actions
                postprocessed_actions = action_dist.sample()
                log_prob = action_dist.log_prob(actions)
                raw_actions = deepcopy(postprocessed_actions)
                for transform in reversed(action_dist.transforms):
                    raw_actions = transform.inv(raw_actions)
                return postprocessed_actions, {
                    'log_prob': log_prob,
                    'raw_action': raw_actions
                }

    def reset(self, obs:Obs = None):
        # mjx policy reset
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)
        self.privileged_obs_history = np.zeros(self.privileged_obs_history_size, dtype=np.float32)
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.is_standing = True
        self.command_list = []
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.step_curr = 0
        print('Resetting...')
        # self.is_prepared = False
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
            print('\nnew command: ', self.fixed_command[5:7])
            if obs.is_done:
                print('Waiting for new observation...')
                self.timer.stop()
                while obs.is_done:
                    msg = self.zmq_receiver.get_msg()
                    if msg is not None and not msg.is_done:
                        obs.is_done = False
                        break
                    time.sleep(0.1)
                self.timer.start()
        print('Reset done!')

    def is_done(self, obs: Obs) -> bool:
        # TODO: any more metric for done?
        return obs.is_done
    
    def rollout_sim(self):
        obs = self.sim.reset()
        start_time = time.time()
        step_curr = 0
        last_action = np.zeros(self.num_action)
        command = self.fixed_command
        print('Rollout sim with command: ', command)

        self.sim.init_recording()
        while obs is not None and not self.is_done(obs) and step_curr < self.finetune_cfg.eval_rollout_length:
            obs.time -= start_time
            time_curr = step_curr * self.control_dt
            phase_signal = self.get_phase_signal(time_curr)
            obs_arr, privileged_obs_arr = self.get_obs(obs, command, phase_signal, last_action)
            action, _ = self.get_action(obs_arr, deterministic=True, is_real=False)
            action_target = self.default_action + self.action_scale * action

            motor_target = self.default_motor_pos.copy()
            motor_target[self.action_mask] = action_target

            motor_target = np.clip(
                motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
            )
            motor_angles: Dict[str, float] = {}
            for motor_name, motor_angle in zip(self.robot.motor_ordering, motor_target):
                motor_angles[motor_name] = motor_angle

            self.sim.set_motor_target(motor_angles)
            self.sim.step()
            obs = self.sim.get_observation()
            step_curr += 1
            last_action = action

        self.sim.save_recording(self.exp_folder, self.sim.dt, cameras=["perspective"])
        print(f'Rollout sim for {step_curr} steps')
        return self.is_done(obs)

    def step(self, obs:Obs, is_real:bool = True):

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
        self.control_inputs = {}
        lin_vel, ang_vel, euler, _ = self.get_tracking_data()
        # print('ang_vel:', ang_vel - obs.ang_vel, 'euler:', euler - obs.euler)
        if msg is not None:
            control_inputs = msg.control_inputs
            obs.ee_force = msg.arm_force
            obs.ee_torque = msg.arm_torque
            obs.arm_ee_pos = msg.arm_ee_pos
            obs.lin_vel = msg.lin_vel + lin_vel
            # obs.ang_vel = ang_vel
            # obs.euler = euler
            obs.is_done = msg.is_done
            # print("control inputs:", control_inputs)
            if msg.is_stopped:
                self.stopped = True
                print("Stopped!")
                return {}, np.zeros(self.num_action)
        else:
            obs.is_done = False
        # import ipdb; ipdb.set_trace()
        if len(control_inputs) == 0:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        self.phase_signal = self.get_phase_signal(time_curr) # TODO: should this similar to obs.time?
        obs_arr, privileged_obs_arr = self.get_obs(obs, command)
        # import ipdb; ipdb.set_trace()
        
        action, _ = self.get_action(obs_arr, deterministic=True, is_real=is_real)

        if msg is not None:
            # print(obs.lin_vel, obs.euler)
            reward_dict = self._compute_reward(obs, action)

            reward = sum(reward_dict.values()) * self.control_dt # TODO: verify, why multiply by dt?
            self.logger.log_step(reward_dict, obs, reward=reward)

            # TODO: last_obs initial value is all None
            if len(control_inputs) > 0 and (control_inputs['walk_x'] != 0 or control_inputs['walk_y'] != 0):
                self.replay_buffer.store(obs_arr, privileged_obs_arr, action, reward, self.is_done(obs))
            # self.replay_buffer.store(obs_arr, privileged_obs_arr, action, reward, self.is_done(obs))

            if len(self.replay_buffer) % 400 == 0:
                print(f"Data size: {len(self.replay_buffer)}, Steps: {self.total_steps}, Fps: {self.total_steps / self.timer.elapsed()}")
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()
            if (len(self.replay_buffer) + 1) % self.finetune_cfg.update_interval == 0:
                self.timer.stop()
                self.zmq_sender.send_msg(ZMQMessage(time=time.time(), is_stopped=True))
                self.logger.plot_queue.put((self.logger.plot_rewards, [])) # no-blocking plot
                # import ipdb; ipdb.set_trace()
                self.replay_buffer.compute_return(self.finetune_cfg.gamma)
                for _ in range(self.finetune_cfg.abppo_update_steps):
                    self.abppo_offline_learner.update(self.replay_buffer)
                self.logger.plot_queue.put((self.logger.plot_updates, [])) # no-blocking plot
                self.logger.print_profiling_data()
                is_sim_done = self.rollout_sim()
                if is_sim_done:
                    import ipdb; ipdb.set_trace()
                self.zmq_sender.send_msg(ZMQMessage(time=time.time(), is_stopped=False))
                self.timer.start()
                # self.reset(obs)

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
        self.total_steps += 1
        self.timer.start()
        self.control_inputs = control_inputs
        # super_obs_arr, delayed_action_jax = super().step(obs, is_real)
        # if not np.allclose(self.last_action, delayed_action, atol=0.1):
        #     import ipdb; ipdb.set_trace()
        return control_inputs, motor_target
    

    def _init_reward(self) -> None:
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        reward_scale_dict = asdict(self.finetune_cfg.finetune_reward_scales)
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

        self.healthy_z_range = self.finetune_cfg.finetune_rewards.healthy_z_range
        self.tracking_sigma = self.finetune_cfg.finetune_rewards.tracking_sigma
        self.arm_force_z_sigma = self.finetune_cfg.finetune_rewards.arm_force_z_sigma
        self.arm_force_y_sigma = self.finetune_cfg.finetune_rewards.arm_force_y_sigma

    def _compute_reward(
        self, obs: Obs, action: np.ndarray
    ):

        reward_dict: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self.reward_names):
            # import ipdb; ipdb.set_trace()
            reward_dict[name] = self.reward_functions[i](obs, action) * self.reward_scales[i]

        return reward_dict
    
    # @njit
    def _reward_torso_pos(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_pos = obs.pos[:2] # TODO: no torso pos
        torso_pos_ref = self.motion_ref[:2]
        error = np.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = np.exp(-200.0 * error**2) # TODO: scale
        return reward
    
    # TODO: change all rotation apis
    def _reward_torso_quat(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_euler = obs.euler
        torso_quat = euler2quat(torso_euler)
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
        reward = np.exp(-20.0 * (angle_diff**2)) # DISCUSS: angle_diff = 3, dot_product = -0.03, result super small
        return reward

    def _reward_lin_vel_xy(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[:2] # TODO: rotate to local? or get it from treadmill
        # array([-0.00291435, -0.00068869, -0.00109268])
        # TODO: verify where we get lin vel from
        # TODO: change treadmill speed according to force x, or estimate from IMU + joint_position
        # TODO: compare which is better
        lin_vel_ref = self.state_ref[7:9]
        # print('lin_vel_ref', lin_vel_ref)
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
        # DISCUSS: array([-2.9682509e-28,  3.4297700e-28,  4.7041364e-28], dtype=float32), very small, reward near 1 ~0.1~1.0
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
        """DISCUSS: 
        motor_pos: aray([ 0.1503303 ,  0.        ,  0.        , -0.5338259 ,  0.        ,
       -0.38042712, -0.15033007, -0.00153399,  0.        ,  0.53535914,
        0.        ,  0.37735915], dtype=float32)
        motor_pos_ref: array([ 0.12043477,  0.00283779, -0.        , -0.52191615,  0.00283779,
       -0.40148139, -0.12043477, -0.00283779, -0.        ,  0.52191615,
        0.00283779,  0.40148139])
        reward: -2e-4
        """
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
    
    def _reward_leg_action_acc(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: store last last action?
        """Reward for tracking action accelerations"""
        error = np.square(action - 2 * self.last_action + self.last_last_action)
        reward = -np.mean(error)
        return reward
    
    # def _reward_feet_contact(self, obs: Obs, action: np.ndarray) -> np.ndarray:
    # def _reward_collision(self, obs: Obs, action: np.ndarray) -> np.ndarray:

    def _reward_survival(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        is_done = self.is_done(obs)
        reward = -np.where(is_done, 1.0, 0.0)
        return reward
    
    def _reward_arm_force_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # import ipdb; ipdb.set_trace()
        ee_force_z = obs.ee_force[2]
        reward = np.exp(-self.arm_force_z_sigma * np.abs(ee_force_z))
        return reward
    
    def _reward_arm_force_y(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ee_force_y = obs.ee_force[1]
        reward = np.exp(-self.arm_force_y_sigma * np.abs(ee_force_y))
        return reward