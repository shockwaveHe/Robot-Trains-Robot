import os
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import toddlerbot.finetuning.networks as networks
from toddlerbot.finetuning.abppo import (
    ABPPO_Offline_Learner,
    AdaptiveBehaviorProximalPolicyOptimization,
)
from toddlerbot.finetuning.dynamics import BaseDynamics, DynamicsNetwork
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.networks import (
    FiLMLayer,
    load_jax_params,
    load_jax_params_into_pytorch,
    load_rsl_params_into_pytorch,
)
from toddlerbot.finetuning.ppo import PPO
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer, RemoteReplayBuffer
from toddlerbot.finetuning.server_client import RemoteClient
from toddlerbot.finetuning.utils import CONST_EPS, Timer
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import (
    euler2mat,
    euler2quat,
    exponential_moving_average,
    interpolate_action,
)
from toddlerbot.utils.misc_utils import log  # , profile

try:
    from pyvicon_datastream import tools
except ImportError:
    pass


class MJXFinetunePolicy(MJXPolicy, policy_name="finetune"):
    def __init__(
        self,
        name,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpts: List[str],
        ip: str,
        eval_mode: bool,
        joystick,
        fixed_command,
        env_cfg,
        finetune_cfg: FinetuneConfig,
        is_real: bool = True,
        *args,
        **kwargs,
    ):
        # set these before super init
        self.eval_mode = eval_mode
        self.is_stopped = False
        self.finetune_cfg: FinetuneConfig = finetune_cfg

        self.num_privileged_obs_history = self.finetune_cfg.frame_stack
        self.privileged_obs_size = self.finetune_cfg.num_single_privileged_obs
        self.privileged_obs_history_size = (
            self.privileged_obs_size * self.num_privileged_obs_history
        )

        super().__init__(
            name,
            robot,
            init_motor_pos,
            "",
            joystick,
            fixed_command,
            env_cfg,
            need_warmup=False,
            *args,
            **kwargs,
        )
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
        self.command_range = np.array(
            self.cfg.commands.command_range
        )  # TODO: set command range to x>0, y~0 and z=0, turn_change = 0
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
        self.last_action_target = self.default_action
        self.is_real = is_real

        print(
            f"Observation size: {self.obs_size}, Privileged observation size: {self.privileged_obs_size}"
        )

        if self.finetune_cfg.update_mode == "local":
            self.replay_buffer = OnlineReplayBuffer(
                self.device,
                self.obs_size * self.num_obs_history,
                self.privileged_obs_size * self.num_privileged_obs_history,
                self.num_action,
                self.finetune_cfg.buffer_size,
                enlarge_when_full=self.finetune_cfg.update_interval
                * self.finetune_cfg.enlarge_when_full,
                validation_size=self.finetune_cfg.buffer_valid_size,
            )
            self.remote_client = None
        else:
            assert self.finetune_cfg.update_mode == "remote"
            self.remote_client = RemoteClient(
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

        # import pickle
        # with open('buffer_mock.pkl', 'rb') as f:
        #     self.replay_buffer: OnlineReplayBuffer = pickle.load(f)
        self._make_networks(
            observation_size=self.finetune_cfg.frame_stack * self.obs_size,
            privileged_observation_size=self.finetune_cfg.frame_stack
            * self.privileged_obs_size,
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
                "toddlerbot", "policies", "checkpoints", "walk_policy"
            )
        print(f"Loading pretrained model from {policy_path}")
        jax_params = load_jax_params(policy_path)
        load_jax_params_into_pytorch(self.policy_net, jax_params[1]["params"])

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
            self._init_tracker()
        else:
            self.zmq_receiver = None
            self.zmq_sender = None
            self.tracker = None

        self.logger = FinetuneLogger(self.exp_folder)

        self.is_paused = False
        self.total_steps = 0
        self.num_updates = 0
        self.timer = Timer()

        self._init_reward()
        self._make_learners()

        self.need_reset = True
        self.learning_stage = "offline"

        self.sim = MuJoCoSim(robot, vis_type=self.finetune_cfg.sim_vis_type, n_frames=1)
        self.min_y_feet_dist = self.finetune_cfg.finetune_rewards.min_feet_y_dist
        self.max_y_feet_dist = self.finetune_cfg.finetune_rewards.max_feet_y_dist

        if len(ckpts) > 0:
            self.load_ckpts(ckpts)

        if self.is_real:
            input("press ENTER to start")

    def load_ckpts(self, ckpts):
        for ckpt in ckpts:
            self.load_networks(ckpt, data_only=False)
            # self.recalculate_reward()

        # if len(self.replay_buffer):
        #     org_policy_net = deepcopy(self.policy_net)
        #     self.logger.plot_queue.put(
        #         (self.logger.plot_rewards, [])
        #     )  # no-blocking plot

        #     for _ in range(self.finetune_cfg.abppo_update_steps):
        #         self.offline_abppo_learner.update(self.replay_buffer)

        #     # import ipdb; ipdb.set_trace()
        #     self.policy_net.load_state_dict(self.abppo._policy_net.state_dict())
        #     assert not torch.allclose(
        #         org_policy_net.mlp.layers[0].weight,
        #         self.policy_net.mlp.layers[0].weight,
        #     )
        #     self.logger.plot_queue.put(
        #         (self.logger.plot_updates, [])
        #     )  # no-blocking plot

    def close(self):
        if hasattr(self, "sim"):
            self.sim.close()
        self.logger.close()
        self.zmq_receiver.close()
        if self.finetune_cfg.update_mode == "local":
            save_networks = input("Save networks? y/n:")
            while save_networks not in ["y", "n"]:
                save_networks = input("Save networks? y/n:")
            if save_networks == "y":
                self.save_networks()
            save_buffer = input("Save replay buffer? y/n:")
            if save_buffer == "y":
                self.replay_buffer.save_compressed(self.exp_folder)

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
        # autoencoder_cfg = self.autoencoder_config

        self.policy_net = networks.GaussianPolicyNetwork(
            observation_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=policy_hidden_layer_sizes,
            action_size=action_size,
            activation_fn=activation_fn,
            use_tanh=self.finetune_cfg.use_tanh,
            noise_std_type=self.finetune_cfg.noise_std_type,
        ).to(self.inference_device)
        self.policy_net_opt = (
            torch.compile(self.policy_net) if self.is_real else self.policy_net
        )
        # Create value network
        self.value_net = networks.ValueNetwork(
            observation_size=privileged_observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn,
        ).to(self.device)

        Q_net_cls = (
            networks.DoubleQNetwork
            if self.finetune_cfg.use_double_q
            else networks.QNetwork
        )
        self.Q_net = Q_net_cls(
            observation_size=privileged_observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn,
        ).to(self.device)

        self.dynamics_net = DynamicsNetwork(
            observation_size=privileged_observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=value_hidden_layer_sizes,
            activation_fn=activation_fn,
        ).to(self.device)

        self.dynamics = BaseDynamics(self.device, self.dynamics_net, self.finetune_cfg)

    def _make_residual_policy(self):
        self.base_policy_net = deepcopy(self.policy_net)
        # make the last layer of the residual policy network to have zero weights, we will only tune the residual network
        # if not self.finetune_cfg.use_latent:
        self.policy_net.mlp.layers[-1].weight.data = torch.zeros_like(
            self.policy_net.mlp.layers[-1].weight
        )
        self.policy_net.mlp.layers[-1].bias.data = torch.zeros_like(
            self.policy_net.mlp.layers[-1].bias
        )
        self.base_policy_net.requires_grad_(False)
        self.base_policy_net.eval()
        self.base_policy_net_opt = (
            torch.compile(self.base_policy_net)
            if self.is_real
            else self.base_policy_net
        )

    def _init_tracker(self):
        self.tracker = tools.ObjectTracker(self.finetune_cfg.vicon_ip)
        self.mocap_marker_offset = self.finetune_cfg.mocap_marker_offset
        self.tracking_alpha = self.finetune_cfg.tracking_alpha
        self.tracking_tf_matrix = self.finetune_cfg.tracking_tf_matrix

        try:
            for _ in range(100):
                result = self.tracker.get_position(self.finetune_cfg.object_name)
                prev_euler = np.array(result[2][0][5:8])
        except Exception as e:
            print(f"Error in initializing the tracker: {e}")
            self.tracker = None
            prev_euler = np.zeros(3)

        self.R_default = euler2mat(prev_euler)
        self.init_euler = prev_euler.copy()
        self.prev_unwrapped = prev_euler.copy()
        self.prev_lin_vel = np.zeros(3)
        self.prev_ang_vel = np.zeros(3)

    def get_tracking_data(self):
        if self.tracker is None:
            log("No tracker available", header="Tracker", level="warning")
            return np.zeros(3), time.time()

        result = self.tracker.get_position(self.finetune_cfg.object_name)
        if not result or len(result[2]) == 0:
            for _ in range(10):
                result = self.tracker.get_position(self.finetune_cfg.object_name)
                if result and len(result[2]) > 0:
                    break
                time.sleep(0.001)
            if not result or len(result[2]) == 0:
                log("No object found in the tracker", header="Tracker", level="warning")
                return np.zeros(3), time.time()
        current_time = time.time()
        current_pos = np.array(result[2][0][2:5]) / 1000
        current_euler = np.array(result[2][0][5:8])

        # Calculate rotation and offset
        R_current = euler2mat(current_euler)
        offset_current = R_current @ self.R_default.T @ self.mocap_marker_offset
        current_pos -= offset_current

        # Calculate time difference
        if hasattr(self, "prev_time"):
            dt = current_time - self.prev_time
        else:
            self.prev_time = current_time
            return np.zeros(3), current_time

        # Calculate linear velocities (x, y, z)
        lin_vel = (
            (current_pos - self.prev_pos) / dt
            if hasattr(self, "prev_pos")
            else np.zeros(3)
        )

        # Update previous values
        self.prev_time = current_time
        self.prev_pos = current_pos

        # EMA filtering
        lin_vel = (
            self.tracking_alpha * lin_vel
            + (1 - self.tracking_alpha) * self.prev_lin_vel
        )
        self.prev_lin_vel = lin_vel

        # Transform velocities
        if self.tracking_tf_matrix is not None:
            lin_vel = self.tracking_tf_matrix @ lin_vel

        return lin_vel, current_time

    def save_networks(self, suffix=""):
        policy_path = os.path.join(self.exp_folder, "policy")

        os.makedirs(policy_path, exist_ok=True)
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(policy_path, f"policy_net{suffix}.pth"),
        )
        torch.save(
            self.value_net.state_dict(),
            os.path.join(policy_path, f"value_net{suffix}.pth"),
        )
        torch.save(
            self.Q_net.state_dict(), os.path.join(policy_path, f"Q_net{suffix}.pth")
        )
        torch.save(
            self.dynamics_net.state_dict(),
            os.path.join(policy_path, f"dynamics_net{suffix}.pth"),
        )
        # torch.save(
        #     {"latent_z": self.online_ppo_learner.latent_z},
        #     os.path.join(policy_path, f"latent_z{suffix}.pt"),
        # )
        self.logger.save_state(self.exp_folder)

    def load_networks(self, exp_folder, data_only=True, suffix="_best"):
        policy_path = os.path.join(exp_folder, "policy")
        buffer_path = os.path.join(exp_folder, "buffer.npz")
        assert (os.path.exists(policy_path) and not data_only) or os.path.exists(
            buffer_path
        )

        if os.path.exists(policy_path) and not data_only:
            org_policy_net = deepcopy(self.policy_net)
            self.policy_net.load_state_dict(
                torch.load(os.path.join(policy_path, f"policy_net{suffix}.pth"))
            )
            self.value_net.load_state_dict(
                torch.load(os.path.join(policy_path, f"value_net{suffix}.pth"))
            )
            self.Q_net.load_state_dict(
                torch.load(os.path.join(policy_path, f"Q_net{suffix}.pth"))
            )
            self.dynamics_net.load_state_dict(
                torch.load(os.path.join(policy_path, f"dynamics_net{suffix}.pth"))
            )
            if torch.allclose(
                org_policy_net.mlp.layers[0].weight,
                self.policy_net.mlp.layers[0].weight,
            ):
                log(
                    "Policy network parameters not changed",
                    header="Networks",
                    level="warning",
                )
            # self.logger.load_state(os.path.join(exp_folder, "logger.pkl"))
            print(f"Loaded pretrained model from {policy_path}")

        if not self.eval_mode and os.path.exists(
            os.path.join(exp_folder, "buffer.npz")
        ):
            try:
                self.replay_buffer.load_compressed(exp_folder)
                # import ipdb; ipdb.set_trace()
                print(f"Loaded replay buffer from {exp_folder}")
            except Exception as e:
                print(f"Error loading replay buffer: {e}")

    def _make_learners(self):
        """Make PPO learners with a PyTorch implementation."""
        self.abppo = AdaptiveBehaviorProximalPolicyOptimization(
            self.device,
            self.policy_net,
            self.finetune_cfg,
        )
        self.offline_abppo_learner = ABPPO_Offline_Learner(
            self.device,
            self.finetune_cfg,
            self.abppo,
            self.Q_net,
            self.value_net,
            self.dynamics,
            self.logger,
        )
        self.online_ppo_learner = PPO(
            self.device,
            self.finetune_cfg,
            self.policy_net,
            self.value_net,
            self.logger,
            self.base_policy_net if self.finetune_cfg.use_residual else None,
            # use_latent=self.finetune_cfg.use_latent,
            # optimize_z=self.finetune_cfg.optimize_z,
            # optimize_critic=self.finetune_cfg.optimize_critic,
            # autoencoder_cfg=self.autoencoder_config,
        )

    def sample_walk_command(self):
        # Sample random angles uniformly between 0 and 2*pi
        theta = self.rng.uniform(low=0, high=2 * np.pi, size=(1,))
        # Parametric equation of ellipse
        x_max = np.where(
            np.sin(theta) > 0, self.command_range[5][1], -self.command_range[5][0]
        )
        x = self.rng.uniform(low=self.deadzone, high=x_max, size=(1,)) * np.sin(theta)
        y_max = np.where(
            np.cos(theta) > 0, self.command_range[6][1], -self.command_range[6][0]
        )
        y = self.rng.uniform(low=self.deadzone, high=y_max, size=(1,)) * np.cos(theta)
        z = np.zeros(1)
        return np.concatenate([x, y, z])

    def sample_turn_command(self):
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

    def _sample_command(self, last_command: Optional[np.ndarray] = None) -> np.ndarray:
        # Randomly sample an index from the command list
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self.rng.uniform(
                low=self.command_range[:5, 0], high=self.command_range[:5, 1], size=(5,)
            )
            pose_command[:5] = 0.0  # TODO: Bring the random pose sampling back

        # def sample_walk_command():
        #     # Sample random angles uniformly between 0 and 2*pi
        #     theta = self.rng.uniform(low=0, high=2 * np.pi, size=(1,))
        #     # Parametric equation of ellipse
        #     x_max = np.where(
        #         np.sin(theta) > 0, self.command_range[5][1], -self.command_range[5][0]
        #     )
        #     x = self.rng.uniform(low=self.deadzone, high=x_max, size=(1,)) * np.sin(
        #         theta
        #     )
        #     y_max = np.where(
        #         np.cos(theta) > 0, self.command_range[6][1], -self.command_range[6][0]
        #     )
        #     y = self.rng.uniform(low=self.deadzone, high=y_max, size=(1,)) * np.cos(
        #         theta
        #     )
        #     z = np.zeros(1)
        #     return np.concatenate([x, y, z])

        # def sample_turn_command():
        #     x = np.zeros(1)
        #     y = np.zeros(1)
        #     z = np.where(
        #         self.rng.uniform((1,)) < 0.5,
        #         self.rng.uniform(
        #             low=self.deadzone,
        #             high=self.command_range[7][1],
        #             size=(1,),
        #         ),
        #         -self.rng.uniform(
        #             low=self.deadzone,
        #             high=-self.command_range[7][0],
        #             size=(1,),
        #         ),
        #     )
        #     return np.concatenate([x, y, z])

        random_number = self.rng.uniform((1,))
        walk_command = np.where(
            random_number < self.zero_chance,
            np.zeros(3),
            np.where(
                random_number < self.zero_chance + self.turn_chance,
                self.sample_turn_command(),
                self.sample_walk_command(),
            ),
        )
        command = np.concatenate([pose_command, walk_command])
        command = np.array([0, 0, 0, 0, 0, 0.2, 0, 0])
        return command

    def get_obs(
        self, obs: Obs, command: np.ndarray, phase_signal=None, last_action=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        # state_ref_ds = np.asarray(self.motion_ref.get_state_ref_ds(self.state_ref, 0.0, command))
        self.state_ref = np.asarray(
            self.motion_ref.get_state_ref(self.state_ref, 0.0, command)
        )

        obs_arr = np.concatenate(
            [
                self.phase_signal if phase_signal is None else phase_signal,  # (2, )
                command[self.command_obs_indices],  # (3, )
                motor_pos_delta * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel * self.obs_scales.dof_vel,  # (30, )
                self.last_action if last_action is None else last_action,  # (12, )
                # motor_pos_error,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel,  # (3, )
                obs.euler * self.obs_scales.euler,  # (3, )
            ]
        )
        privileged_obs_arr = np.concatenate(
            [
                self.phase_signal if phase_signal is None else phase_signal,  # (2, )
                command[self.command_obs_indices],  # (3, )
                motor_pos_delta * self.obs_scales.dof_pos,  # (30, )
                obs.motor_vel * self.obs_scales.dof_vel,  # (30, )
                self.last_action if last_action is None else last_action,  # (12, )
                obs.motor_pos,  # (30, ) change from motor_pos_error
                obs.lin_vel * self.obs_scales.lin_vel,  # (3, )
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

    @torch.no_grad()
    def get_action(
        self, obs_arr: np.ndarray, deterministic: bool = True, is_real: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # import ipdb; ipdb.set_trace()
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32).squeeze(0)
        # latent_tensor = self.online_ppo_learner.get_latent()
        # TODO: change name to reuse _opt networks
        action_dist = self.policy_net_opt(obs_tensor.to(self.inference_device))
        # action_dist = self.policy_net_opt(
        #     obs_tensor.to(self.inference_device),
        #     latent_tensor.to(self.inference_device)
        #     if latent_tensor is not None
        #     else None,
        # )

        if deterministic:
            # Deterministic: use mode
            if isinstance(action_dist, torch.distributions.TransformedDistribution):
                actions_pi = action_dist.base_dist.mode
                for transform in action_dist.transforms:
                    actions_pi = transform(actions_pi)
            else:
                assert isinstance(action_dist, torch.distributions.Normal)
                actions_pi = action_dist.mean
        else:
            # Stochastic: sample raw pre-tanh actions
            actions_pi = (
                action_dist.sample()
            )  # action is transformed so no need to clamp
        log_prob = action_dist.log_prob(actions_pi).sum()

        if self.finetune_cfg.use_residual:
            base_action_dist = self.base_policy_net_opt(
                obs_tensor.to(self.inference_device)
            )
            if isinstance(
                base_action_dist, torch.distributions.TransformedDistribution
            ):
                base_actions = base_action_dist.base_dist.mode
                for transform in base_action_dist.transforms:
                    base_actions = transform(base_actions)
            else:
                assert isinstance(base_action_dist, torch.distributions.Normal)
                base_actions = base_action_dist.mean
            actions_real = self._residual_action_scale * actions_pi + base_actions
        else:
            actions_real = actions_pi
        actions_real.clamp_(-1.0 + CONST_EPS, 1.0 - CONST_EPS)
        return (
            actions_pi.cpu().numpy().flatten(),
            actions_real.cpu().numpy().flatten(),
            log_prob.cpu().numpy().flatten(),
        )

    def reset(self, obs: Obs = None):
        # mjx policy reset
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)
        self.privileged_obs_history = np.zeros(
            self.privileged_obs_history_size, dtype=np.float32
        )
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.is_standing = True
        self.command_list = []
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.step_curr = 0
        self.last_action = np.zeros(self.num_action)
        self.last_last_action = np.zeros(self.num_action)
        print("Resetting...")
        # self.is_prepared = False
        if obs is not None:
            # TODO: more things to reset?
            path_pos = np.zeros(3)
            # path_yaw = self.rng.uniform(low=0, high=2 * np.pi, size=(1,))
            path_yaw = obs.euler[2]  # TODO: only change in a small range
            print(f"Resetting with yaw: {path_yaw}")
            path_euler = np.array([0.0, 0.0, np.degrees(path_yaw)])
            path_quat = euler2quat(path_euler)  # TODO: verify usage, wxyz or xyzw?
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
            # motor_pos = obs.joint_pos[self.q_start_idx + self.motor_indices]
            # joint_pos = obs.joint_pos[self.q_start_idx + self.joint_indices]
            motor_pos = np.zeros_like(self.default_motor_pos)
            joint_pos = np.zeros_like(self.default_joint_pos)
            stance_mask = np.ones(2)

            state_ref = np.concatenate(
                [
                    path_pos,
                    path_quat,
                    lin_vel,
                    ang_vel,
                    motor_pos,
                    joint_pos,
                    stance_mask,
                ]
            )
            self.fixed_command = self._sample_command()
            self.state_ref = np.asarray(
                self.motion_ref.get_state_ref(state_ref, 0.0, self.fixed_command)
            )
            print("\nnew command: ", self.fixed_command[5:7])
            if obs.is_done:
                print("Waiting for new observation...")
                self.timer.stop()
                while obs.is_done:
                    msg = self.zmq_receiver.get_msg()
                    if msg is not None and not msg.is_done:
                        obs.is_done = False
                        break
                    time.sleep(0.1)
                self.timer.start()
        print("Reset done!")

    def is_done(self, obs: Obs) -> bool:
        # TODO: any more metric for done?
        return obs.is_done

    def rollout_sim(self):
        obs = self.sim.reset()
        self.reset(obs)
        start_time = time.time()
        step_curr, total_reward = 0, 0
        last_action = np.zeros(self.num_action)
        command = self.fixed_command
        print("Rollout sim with command: ", command[5:7])

        self.sim.init_recording()
        while (
            obs is not None
            and not self.is_done(obs)
            and step_curr < self.finetune_cfg.eval_rollout_length
        ):
            obs.time -= start_time
            time_curr = step_curr * self.control_dt
            phase_signal = self.get_phase_signal(time_curr)
            obs_arr, privileged_obs_arr = self.get_obs(
                obs, command, phase_signal, last_action
            )
            action_pi, action_real, _ = self.get_action(
                obs_arr, deterministic=True, is_real=False
            )
            obs.state_ref = self.state_ref[:29]
            feet_pos = self.sim.get_feet_pos()
            feet_y_dist = feet_pos["left"][1] - feet_pos["right"][1]
            obs.feet_y_dist = feet_y_dist
            reward_dict = self._compute_reward(obs, action_real)
            total_reward += sum(reward_dict.values()) * self.control_dt

            obs = self.sim.get_observation()
            step_curr += 1
            last_action = action_real

        self.sim.save_recording(self.exp_folder, self.sim.dt, cameras=["perspective"])
        print(f"Rollout sim for {step_curr} steps with reward: {total_reward}")
        if self.is_done(obs):
            print("Sim early terminated!")
            import ipdb

            ipdb.set_trace()
        return total_reward

    def recalculate_reward(self):
        self.logger.reset()
        self.last_action = np.zeros(self.num_action)
        self.last_last_action = np.zeros(self.num_action)
        assert isinstance(self.replay_buffer, OnlineReplayBuffer)
        for i in tqdm(range(len(self.replay_buffer)), desc="Recalculating reward"):
            obs, privileged_obs, action, reward, done, trunc, _, raw_obs = (
                self.replay_buffer[i]
            )
            reward_dict = self._compute_reward(raw_obs, action)
            self.replay_buffer._reward[i] = sum(reward_dict.values()) * self.control_dt
            if not (done or trunc):
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()
            else:
                self.last_last_action = np.zeros(self.num_action)
                self.last_action = np.zeros(self.num_action)
            # self.logger.log_step(
            #     reward_dict,
            #     raw_obs,
            #     reward=reward,
            #     feet_dist=raw_obs.feet_y_dist,
            #     # walk_command=obs[3],
            # )
        self.replay_buffer.compute_return(self.finetune_cfg.gamma)

    def send_step_count(self, obs=None):
        self.zmq_sender.send_msg(
            ZMQMessage(time=time.time(), total_steps=self.total_steps)
        )
        # if obs is None:
        #     self.zmq_sender.send_msg(
        #         ZMQMessage(time=time.time(), total_steps=self.total_steps)
        #     )
        # else:
        #     self.zmq_sender.send_msg(
        #         ZMQMessage(
        #             time=time.time(),
        #             total_steps=self.total_steps,
        #             torso_roll=obs.euler[0],
        #             torso_pitch=obs.euler[1],
        #         )
        #     )

    def update_policy(self):
        self.timer.stop()
        if self.is_real:
            self.zmq_sender.send_msg(ZMQMessage(time=time.time(), is_stopped=True))
        self.logger.plot_queue.put((self.logger.plot_rewards, []))  # no-blocking plot
        # self.logger.plot_queue.put((self.logger.plot_updates, []))  # no-blocking plot
        self.replay_buffer.compute_return(self.finetune_cfg.gamma)
        # org_policy_net = deepcopy(self.policy_net)

        if self.learning_stage == "offline":
            for _ in range(self.finetune_cfg.abppo_update_steps):
                self.offline_abppo_learner.update(self.replay_buffer)
            self.policy_net.load_state_dict(self.abppo._policy_net.state_dict())
        elif self.learning_stage == "online":
            self.online_ppo_learner.update(
                self.replay_buffer,
                self.total_steps - self.finetune_cfg.offline_total_steps,
            )
            self.policy_net.load_state_dict(
                self.online_ppo_learner._policy_net.state_dict()
            )

        # assert not torch.allclose(
        #     org_policy_net.mlp.layers[0].weight,
        #     self.policy_net.mlp.layers[0].weight,
        # )
        self.logger.plot_queue.put((self.logger.plot_updates, []))  # no-blocking plot
        # if (
        #     getattr(self, "state_ref", None) is not None
        # ):  # hack to only rollout in walk tasks. Yao: Don't know why this is commented in walk
        #     self.rollout_sim()
        if self.is_real:
            self.need_reset = True
            self.zmq_sender.send_msg(ZMQMessage(time=time.time(), is_stopped=False))
        self.timer.start()

    def switch_learning_stage(self):
        # input('switch stage?')
        if self.finetune_cfg.update_mode == "local" and len(self.replay_buffer) > 0:
            # self.update_policy()
            save_offline_buffer = input("Save offline buffer? y/n:")
            while save_offline_buffer not in ["y", "n"]:
                save_offline_buffer = input("Save offline buffer? y/n:")
            if save_offline_buffer == "y":
                self.replay_buffer.save_compressed(self.exp_folder)
            self.offline_abppo_learner.fit_q_v(self.replay_buffer)
        self.replay_buffer.reset()
        self.learning_stage = "online"
        self.online_ppo_learner.set_networks(self.value_net, self.policy_net)
        assert torch.allclose(
            self.value_net.mlp.layers[0].weight,
            self.offline_abppo_learner._value_net.mlp.layers[0].weight,
        )
        print("Switched to online learning!")

    # @profile()
    def step(self, obs: Obs, is_real: bool = True):
        if not self.is_prepared:
            self.traj_start_time = time.time()
            self.is_prepared = True
            self.prep_duration = 5.0 if is_real else 0.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                obs.motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=2.0 if is_real else 0.0,
            )
        time_curr = time.time()
        if time_curr - self.traj_start_time < self.prep_duration:
            motor_target = np.asarray(
                interpolate_action(
                    time_curr - self.traj_start_time, self.prep_time, self.prep_action
                )
            )
            return {}, motor_target, obs

        if self.last_msg is None:
            msg = None
            while self.is_real and msg is None:
                msg = self.zmq_receiver.get_msg()
        else:
            msg = self.zmq_receiver.get_msg()

        if msg is None:
            msg = self.last_msg
        else:
            self.last_msg = msg

        # TODO: remove this for the real world
        # msg = ZMQMessage(
        #     time=time_curr,
        #     control_inputs={"walk_x": 0.1, "walk_y": 0.0, "walk_turn": 0.0},
        #     lin_vel=np.zeros(3, dtype=np.float32),
        #     arm_force=np.zeros(3, dtype=np.float32),
        #     arm_torque=np.zeros(3, dtype=np.float32),
        #     arm_ee_pos=np.zeros(3, dtype=np.float32),
        # )

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

        if self.need_reset:
            self.reset(obs)
            self.need_reset = False

        time_curr = self.step_curr * self.control_dt

        # if self.finetune_cfg.update_mode == "local":
        #     self.sim.set_motor_angles(obs.motor_pos)
        #     self.sim.forward()
        #     feet_pos = self.sim.get_feet_pos()
        #     feet_y_dist = feet_pos["left"][1] - feet_pos["right"][1]
        #     obs.feet_y_dist = feet_y_dist
        control_inputs: Dict[str, float] = {}
        self.control_inputs = {}
        # lin_vel, _ = self.get_tracking_data()
        # print('ang_vel:', ang_vel - obs.ang_vel, 'euler:', euler - obs.euler)
        control_inputs = msg.control_inputs
        obs.ee_force = msg.arm_force
        obs.ee_torque = msg.arm_torque
        obs.arm_ee_pos = msg.arm_ee_pos
        if self.eval_mode:
            obs.lin_vel = np.array([0.2, 0, 0])
            obs.ee_force = np.zeros(3)
            obs.ee_torque = np.zeros(3)
            obs.arm_ee_pos = np.zeros(3)
        else:
            obs.lin_vel = msg.lin_vel  # + lin_vel
        # obs.ang_vel = ang_vel
        # obs.euler = euler
        obs.is_done = msg.is_done
        obs.state_ref = self.state_ref[:29]
        # print("control inputs:", control_inputs)
        if msg.is_stopped:
            self.stopped = True
            print("Stopped!")
            return {}, np.zeros(self.num_action), obs

        # import ipdb; ipdb.set_trace()
        if self.eval_mode or len(control_inputs) == 0:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        self.phase_signal = self.get_phase_signal(time_curr)
        obs_arr, privileged_obs_arr = self.get_obs(obs, command)

        if self.total_steps == self.finetune_cfg.offline_total_steps:
            self.switch_learning_stage()

        if (
            self.remote_client is not None
            and self.remote_client.ready_to_update
            and self.total_steps < self.total_training_steps
        ):
            # import ipdb; ipdb.set_trace()
            self.num_updates += 1
            print(f"Updated policy network to {self.num_updates}!")
            # assert not torch.allclose(
            #     self.policy_net.mlp.layers[0].weight,
            #     self.remote_client.new_state_dict["mlp.layers.0.weight"],
            # )
            if self.finetune_cfg.optimize_z:
                self.online_ppo_learner.latent_z = self.remote_client.new_state_dict[
                    "latent_z"
                ].clone()
            else:
                self.policy_net.load_state_dict(self.remote_client.new_state_dict)

            self.remote_client.ready_to_update = False

        # deterministic = (
        #     self.learning_stage == "offline"
        # )  # use deterministic action during offline learning
        # action_pi, action_real, action_logprob = self.get_action(
        #     obs_arr, deterministic=deterministic, is_real=is_real
        # )
        if self.learning_stage == "online":
            action_pi, action_real, action_logprob = self.get_action(
                obs_arr, deterministic=self.eval_mode, is_real=is_real
            )
            action_real_copy = action_real.copy()
        else:
            action_pi, action_real, action_logprob = self.get_action(
                obs_arr, deterministic=self.eval_mode, is_real=is_real
            )
            action_real_copy = action_real.copy()

        obs.raw_action_mean = action_pi.mean()
        obs.base_action_mean = (action_real - action_pi).mean()
        if msg is not None:
            # print(obs.lin_vel, obs.euler)
            if self.finetune_cfg.update_mode == "local":
                reward_dict = self._compute_reward(obs, action_real)
                self.last_last_action = self.last_action.copy()
                self.last_action = action_real.copy()

                reward = (
                    sum(reward_dict.values()) * self.control_dt
                )  # TODO: verify, why multiply by dt?
                self.logger.log_step(
                    reward_dict,
                    obs,
                    reward=reward,
                    # feet_dist=feet_y_dist,
                    action_pi=action_pi.mean(),
                    action_real=action_real_copy.mean(),
                    walk_command=control_inputs["walk_x"]
                    if not self.eval_mode
                    else self.fixed_command[5],
                )
            else:
                reward = 0.0

            self.reward_list.append(reward)

            time_elapsed = self.timer.elapsed()
            if time_elapsed < self.total_steps * self.control_dt:
                time.sleep(self.total_steps * self.control_dt - time_elapsed)

            if (len(self.replay_buffer) + 1) % 400 == 0:
                print(
                    f"Data size: {len(self.replay_buffer)}, Steps: {self.total_steps}, Fps: {self.total_steps / self.timer.elapsed()}"
                )
            self.send_step_count(obs)
            if (self.eval_mode) or (
                len(control_inputs) > 0
                and (control_inputs["walk_x"] != 0 or control_inputs["walk_y"] != 0)
            ):
                time_to_update = (
                    not self.eval_mode
                    and self.learning_stage == "offline"
                    and (len(self.replay_buffer) + 1)
                    % self.finetune_cfg.update_interval
                    == 0
                ) or (
                    not self.eval_mode
                    and self.learning_stage == "online"
                    and len(self.replay_buffer) == self.finetune_cfg.online.batch_size
                )
                truncated = time_to_update and self.finetune_cfg.update_mode == "local"
                self.replay_buffer.store(
                    obs_arr,
                    privileged_obs_arr,
                    action_pi,
                    reward,
                    self.is_done(obs),
                    truncated or self.is_paused,
                    action_logprob,
                    raw_obs=deepcopy(obs),
                )

                if truncated:
                    reward_epi = np.mean(self.reward_list)
                    self.reward_epi_list.append(reward_epi)
                    if (
                        reward_epi > self.reward_epi_best
                        # and self.external_guidance_stage == "free"
                    ):
                        self.reward_epi_best = reward_epi
                        self.save_networks(suffix="_best")
                        print(f"Best reward: {self.reward_epi_best}")

                    self.update_policy()

        if is_real:
            delayed_action = action_real
        else:
            self.action_buffer = np.roll(self.action_buffer, action_real.size)
            self.action_buffer[: action_real.size] = action_real
            delayed_action = self.action_buffer[-self.num_action :]

        action_target = self.default_action + self.action_scale * delayed_action

        if self.filter_type == "ema":
            action_target = exponential_moving_average(
                self.ema_alpha, action_target, self.last_action_target
            )

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
        return control_inputs, motor_target, obs

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

        # self.healthy_z_range = self.finetune_cfg.finetune_rewards.healthy_z_range
        # self.tracking_sigma = self.finetune_cfg.finetune_rewards.tracking_sigma
        # self.arm_force_z_sigma = self.finetune_cfg.finetune_rewards.arm_force_z_sigma
        # self.arm_force_y_sigma = self.finetune_cfg.finetune_rewards.arm_force_y_sigma
        # self.arm_force_x_sigma = self.finetune_cfg.finetune_rewards.arm_force_x_sigma

    def _compute_reward(self, obs: Obs, action: np.ndarray):
        reward_dict: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self.reward_names):
            # import ipdb; ipdb.set_trace()
            reward_dict[name] = (
                self.reward_functions[i](obs, action) * self.reward_scales[i]
            )

        return reward_dict
