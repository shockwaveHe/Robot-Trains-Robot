import os
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from scipy.spatial.transform import Rotation

import toddlerbot.finetuning.networks as networks
from toddlerbot.finetuning.abppo import (
    ABPPO_Offline_Learner,
    AdaptiveBehaviorProximalPolicyOptimization,
)
from toddlerbot.finetuning.dynamics import BaseDynamics, DynamicsNetwork
from toddlerbot.finetuning.finetune_config import FinetuneConfig, get_finetune_config
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.networks import FiLMLayer, load_rsl_params_into_pytorch
from toddlerbot.finetuning.ppo import PPO
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer, RemoteReplayBuffer
from toddlerbot.finetuning.server_client import RemoteClient
from toddlerbot.finetuning.utils import CONST_EPS, Timer
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import (
    euler2mat,
    euler2quat,
    exponential_moving_average,
    interpolate_action,
)
from toddlerbot.utils.misc_utils import log


class WalkFinetunePolicy(MJXFinetunePolicy, policy_name="walk_finetune"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpts: str = "",
        ip: str = "",
        eval_mode: bool = False,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        exp_folder: Optional[str] = "",
        env_cfg: Optional[Dict] = None,
        finetune_cfg: Optional[Dict] = None,
        is_real: bool = True,
        *args,
        **kwargs,
    ):
        if env_cfg is None:
            if is_real:
                env_cfg = get_env_config("walk_real")
            else:
                env_cfg = get_env_config("walk")
        if finetune_cfg is None:
            finetune_cfg = get_finetune_config("walk", exp_folder)
        self.cycle_time = env_cfg.action.cycle_time
        self.command_discount_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.torso_roll_range = finetune_cfg.finetune_rewards.torso_roll_range
        self.torso_pitch_range = finetune_cfg.finetune_rewards.torso_pitch_range
        self.last_torso_yaw = 0.0
        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = finetune_cfg.finetune_rewards.min_feet_y_dist
        self.max_feet_y_dist = finetune_cfg.finetune_rewards.max_feet_y_dist

        # set these before super init
        self.eval_mode = eval_mode
        self.is_stopped = False
        self.finetune_cfg: FinetuneConfig = finetune_cfg

        self.num_privileged_obs_history = self.finetune_cfg.frame_stack
        self.privileged_obs_size = self.finetune_cfg.num_single_privileged_obs
        self.privileged_obs_history_size = (
            self.privileged_obs_size * self.num_privileged_obs_history
        )

        self.autoencoder_config = None
        if self.finetune_cfg.use_latent:
            with open(
                os.path.join("toddlerbot", "autoencoder", "config.yaml"), "r"
            ) as f:
                autoencoder_config = yaml.safe_load(f)
                # autoencoder_config["data"]["time_str"] = dynamics_time_str
                if autoencoder_config["model"]["dynamics_type"] == "params":
                    autoencoder_config["model"]["num_split"] = 2
                elif autoencoder_config["model"]["dynamics_type"] == "hyper":
                    autoencoder_config["model"]["num_split"] = 1
                self.autoencoder_config = autoencoder_config

        super(MJXFinetunePolicy, self).__init__(
            name,
            robot,
            init_motor_pos,
            "",
            joystick,
            fixed_command,
            env_cfg,
            need_warmup=False,
            exp_folder=exp_folder,
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

        if self.eval_mode:
            self.finetune_cfg.update_mode = "local"

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

        self._make_networks(
            observation_size=self.finetune_cfg.frame_stack * self.obs_size,
            privileged_observation_size=self.finetune_cfg.frame_stack
            * self.privileged_obs_size,
            action_size=self.num_action,
            value_hidden_layer_sizes=self.finetune_cfg.value_hidden_layer_sizes,
            policy_hidden_layer_sizes=self.finetune_cfg.policy_hidden_layer_sizes,
        )

        if len(ckpts) > 0:
            run_name = f"{self.robot.name}_walk_ppo_{ckpts[0]}"
            policy_path = os.path.join("results", run_name, "model_best.pt")
            if os.path.exists(policy_path):
                print(f"Loading pretrained model from {policy_path}")
                # jax_params = load_jax_params(policy_path)
                # load_jax_params_into_pytorch(self.policy_net, jax_params[1]["params"])
                rsl_params = torch.load(os.path.join(policy_path))["model_state_dict"]
                load_rsl_params_into_pytorch(
                    self.policy_net, self.value_net, rsl_params
                )
            else:
                self.load_ckpts(
                    [
                        os.path.join(
                            "results",
                            f"{self.robot.name}_{self.name}_real_world_{ckpts[0]}",
                        )
                    ]
                )

        if self.finetune_cfg.use_latent:
            self.finetune_cfg.use_residual = False

        if self.finetune_cfg.use_residual:
            self._make_residual_policy()
            self._residual_action_scale = self.finetune_cfg.residual_action_scale

        # loading residual policy
        if self.eval_mode:
            try:
                self.load_ckpts(
                    [
                        os.path.join(
                            "results",
                            f"{self.robot.name}_{self.name}_real_world_{ckpts[1]}",
                        )
                    ]
                )
                print("Residual policy loaded successfully")
            except Exception as e:
                print("residual policy not loaded correstly: ", e)

        self.obs_history = np.zeros(self.num_obs_history * self.obs_size)
        self.privileged_obs_history = np.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        if self.is_real:
            self.zmq_receiver = ZMQNode(type="receiver")
            assert len(ip) > 0, "Please provide the IP address of the sender"
            self.zmq_sender = ZMQNode(type="sender", ip=ip)
            # self._init_tracker()
        else:
            self.zmq_receiver = None
            self.zmq_sender = None
            self.tracker = None

        self.logger = FinetuneLogger(self.exp_folder)

        self.is_paused = False
        self.total_steps = 0
        self.total_training_steps = 75000
        self.num_updates = 0
        self.timer = Timer()
        self.last_msg = None
        if self.eval_mode:
            self.last_msg = ZMQMessage(time=time.time())

        self._init_reward()
        self._make_learners()

        self.need_reset = True
        self.learning_stage = "offline"

        self.reward_list = []
        self.reward_epi_list = []
        self.reward_epi_best = -np.inf

    def get_phase_signal(self, time_curr: float):
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.zeros(self.num_commands, dtype=np.float32)
        command[5:] = self.command_discount_factor * np.array(
            [
                control_inputs["walk_x"],
                control_inputs["walk_y"],
                control_inputs["walk_turn"],
            ]
        )

        # print(f"walk_command: {command}")
        return command

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
        autoencoder_cfg = self.autoencoder_config

        film_layers = None
        if autoencoder_cfg is not None:
            latent_dim = (
                autoencoder_cfg["model"]["n_embd"]
                * autoencoder_cfg["model"]["num_splits"]
            )
            self.latent_dim = latent_dim
            film_layers = nn.ModuleList()
            film_layers.append(FiLMLayer(latent_dim, policy_hidden_layer_sizes[0]))
            for i in range(1, len(policy_hidden_layer_sizes)):
                film_layers.append(FiLMLayer(latent_dim, policy_hidden_layer_sizes[i]))

            for film_layer in film_layers:
                nn.init.constant_(film_layer.film.weight, 0.0)
                nn.init.constant_(
                    film_layer.film.bias[: film_layer.film.out_features // 2], 1.0
                )
                nn.init.constant_(
                    film_layer.film.bias[film_layer.film.out_features // 2 :], 0.0
                )

        self.policy_net = networks.GaussianPolicyNetwork(
            observation_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layers=policy_hidden_layer_sizes,
            action_size=action_size,
            activation_fn=activation_fn,
            use_tanh=self.finetune_cfg.use_tanh,
            noise_std_type=self.finetune_cfg.noise_std_type,
            film_layers=film_layers,
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
        if not self.finetune_cfg.use_latent:
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

    def save_networks(self, suffix="_best"):
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
        torch.save(
            {"latent_z": self.online_ppo_learner.latent_z},
            os.path.join(policy_path, f"latent_z{suffix}.pt"),
        )
        self.logger.save_state(self.exp_folder)

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
            use_latent=self.finetune_cfg.use_latent,
            optimize_z=self.finetune_cfg.optimize_z,
            optimize_critic=self.finetune_cfg.optimize_critic,
            autoencoder_cfg=self.autoencoder_config,
        )

    def _sample_command(self, last_command: Optional[np.ndarray] = None) -> np.ndarray:
        # Randomly sample an index from the command list
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self.rng.uniform(
                low=self.command_range[:5, 0], high=self.command_range[:5, 1], size=(5,)
            )
            pose_command[:5] = 0.0  # TODO: Bring the random pose sampling back

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
        command = np.array([0, 0, 0, 0, 0, 0.2, 0, 0])  # tracking command
        return command

    @torch.no_grad()
    def get_action(
        self, obs_arr: np.ndarray, deterministic: bool = True, is_real: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # import ipdb; ipdb.set_trace()
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32).squeeze(0)
        latent_tensor = self.online_ppo_learner.get_latent()
        # TODO: change name to reuse _opt networks
        action_dist = self.policy_net_opt(
            obs_tensor.to(self.inference_device),
            latent_tensor.to(self.inference_device)
            if latent_tensor is not None
            else None,
        )

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

    def send_step_count(self, obs=None):
        if obs is None:
            self.zmq_sender.send_msg(
                ZMQMessage(time=time.time(), total_steps=self.total_steps)
            )
        else:
            self.zmq_sender.send_msg(
                ZMQMessage(
                    time=time.time(),
                    total_steps=self.total_steps,
                    torso_roll=obs.euler[0],
                    torso_pitch=obs.euler[1],
                )
            )

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        # control_inputs, motor_target, obs = super().step(obs, is_real) copied below
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

        control_inputs: Dict[str, float] = {}
        self.control_inputs = {}
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
        obs.is_done = msg.is_done
        obs.state_ref = self.state_ref[:29]
        if msg.is_stopped:
            self.stopped = True
            print("Stopped!")
            return {}, np.zeros(self.num_action), obs

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
            self.num_updates += 1
            print(f"Updated policy network to {self.num_updates}!")
            if self.finetune_cfg.optimize_z:
                self.online_ppo_learner.latent_z = self.remote_client.new_state_dict[
                    "latent_z"
                ].clone()
            else:
                self.policy_net.load_state_dict(self.remote_client.new_state_dict)

            self.remote_client.ready_to_update = False

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
        # end of copying the original step function
        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            all_zeros = all(np.all(command == 0) for command in last_commands)
            self.is_standing = all_zeros and abs(self.phase_signal[0]) > 1 - 1e-6
        else:
            self.is_standing = False
        self.last_torso_yaw = obs.euler[2]
        return control_inputs, motor_target, obs

    def _init_reward(self) -> None:
        super()._init_reward()
        self.healthy_z_range = self.finetune_cfg.finetune_rewards.healthy_z_range
        self.tracking_sigma = self.finetune_cfg.finetune_rewards.tracking_sigma
        self.arm_force_z_sigma = self.finetune_cfg.finetune_rewards.arm_force_z_sigma
        self.arm_force_y_sigma = self.finetune_cfg.finetune_rewards.arm_force_y_sigma
        self.arm_force_x_sigma = self.finetune_cfg.finetune_rewards.arm_force_x_sigma

    def _reward_torso_pos(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_pos = obs.pos[:2]  # TODO: no torso pos
        torso_pos_ref = self.motion_ref[:2]
        error = np.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = np.exp(-200.0 * error**2)  # TODO: scale
        return reward

    # TODO: change all rotation apis
    def _reward_torso_quat(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        torso_euler = obs.euler
        torso_quat = euler2quat(torso_euler)
        path_quat_ref = obs.state_ref[3:7]
        path_rot = Rotation.from_quat(path_quat_ref)

        waist_joint_pos = obs.state_ref[
            self.ref_start_idx + self.robot.nu + self.waist_motor_indices
        ]
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
        reward = np.exp(
            -20.0 * (angle_diff**2)
        )  # DISCUSS: angle_diff = 3, dot_product = -0.03, result super small
        return reward

    def _reward_lin_vel_x(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[0]  # TODO: rotate to local? or get it from treadmill
        # array([-0.00291435, -0.00068869, -0.00109268])
        # TODO: verify where we get lin vel from
        # TODO: change treadmill speed according to force x, or estimate from IMU + joint_position
        # TODO: compare which is better
        lin_vel_ref = obs.state_ref[7]
        # print('lin_vel_ref', lin_vel_ref)
        error = np.abs(lin_vel - lin_vel_ref)
        reward = np.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_lin_vel_y(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[1]  # TODO: rotate to local? or get it from treadmill
        # array([-0.00291435, -0.00068869, -0.00109268])
        # TODO: verify where we get lin vel from
        # TODO: change treadmill speed according to force x, or estimate from IMU + joint_position
        # TODO: compare which is better
        lin_vel_ref = obs.state_ref[8]
        # print('lin_vel_ref', lin_vel_ref)
        error = np.abs(lin_vel - lin_vel_ref)
        reward = np.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_lin_vel_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        lin_vel = obs.lin_vel[2]  # TODO: change to normal force
        lin_vel_ref = obs.state_ref[9]
        error = np.abs(lin_vel - lin_vel_ref)
        reward = np.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_ang_vel_x(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # DISCUSS: array([-2.9682509e-28,  3.4297700e-28,  4.7041364e-28], dtype=float32), very small, reward near 1 ~0.1~1.0
        ang_vel = obs.ang_vel[0]
        ang_vel_ref = obs.state_ref[10]
        error = np.abs(ang_vel - ang_vel_ref)
        reward = np.exp(-self.tracking_sigma / 4 * error**2)
        return reward

    def _reward_ang_vel_y(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # DISCUSS: array([-2.9682509e-28,  3.4297700e-28,  4.7041364e-28], dtype=float32), very small, reward near 1 ~0.1~1.0
        ang_vel = obs.ang_vel[1]
        ang_vel_ref = obs.state_ref[11]
        error = np.abs(ang_vel - ang_vel_ref)
        reward = np.exp(-self.tracking_sigma / 4 * error**2)
        return reward

    def _reward_ang_vel_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ang_vel = obs.ang_vel[2]
        ang_vel_ref = obs.state_ref[12]
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
        print(self.ref_start_idx + self.leg_motor_indices)
        motor_pos_ref = obs.state_ref[self.ref_start_idx + self.leg_motor_indices]
        error = motor_pos - motor_pos_ref
        reward = -np.mean(error**2)  # TODO: why not exp?
        return reward

    # def _reward_motor_torque(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: how to get motor torque?
    # def _reward_energy(self, obs: Obs, action: np.ndarray) -> np.ndarray: # TODO: how to get energy?

    def _reward_leg_action_rate(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        error = np.square(action - self.last_action)
        reward = -np.mean(error)
        return reward

    def _reward_leg_action_acc(
        self, obs: Obs, action: np.ndarray
    ) -> np.ndarray:  # TODO: store last last action?
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

    def _reward_arm_force_x(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ee_force_x = obs.ee_force[0]
        reward = np.exp(-self.arm_force_x_sigma * np.abs(ee_force_x))
        return reward

    def _reward_arm_force_y(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        ee_force_y = obs.ee_force[1]
        reward = np.exp(-self.arm_force_y_sigma * np.abs(ee_force_y))
        return reward

    def _reward_arm_force_z(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # import ipdb; ipdb.set_trace()
        ee_force_z = obs.ee_force[2]
        reward = np.exp(-self.arm_force_z_sigma * np.abs(ee_force_z))
        return reward

    def _reward_torso_roll(self, obs: Obs, action: np.ndarray) -> float:
        """Reward for torso pitch"""
        torso_roll = obs.euler[0]
        # DISCUSS: torso_roll = -0.03, min and max are all 0.
        roll_min = np.clip(
            torso_roll - self.torso_roll_range[0], a_min=-np.inf, a_max=0.0
        )
        roll_max = np.clip(
            torso_roll - self.torso_roll_range[1], a_min=0.0, a_max=np.inf
        )
        reward = (np.exp(-np.abs(roll_min) * 100) + np.exp(-np.abs(roll_max) * 100)) / 2
        return reward

    def _reward_torso_pitch(self, obs: Obs, action: np.ndarray) -> float:
        """Reward for torso pitch"""
        torso_pitch = obs.euler[1]
        # DISCUSS: torso_pitch = 0.05, min and max are all 0.
        pitch_min = np.clip(
            torso_pitch - self.torso_pitch_range[0], a_min=-np.inf, a_max=0.0
        )
        pitch_max = np.clip(
            torso_pitch - self.torso_pitch_range[1], a_min=0.0, a_max=np.inf
        )
        reward = (
            np.exp(-np.abs(pitch_min) * 100) + np.exp(-np.abs(pitch_max) * 100)
        ) / 2
        return reward

    def _reward_torso_yaw_vel(self, obs: Obs, action: np.ndarray) -> float:
        """Reward for torso yaw velocity"""
        torso_yaw_vel = obs.ang_vel[2]
        reward = -np.abs(torso_yaw_vel)
        return reward

    # def _reward_feet_air_time(self, obs: Obs, action: np.ndarray) -> float:
    #     # Reward air time.
    #     contact_filter = np.logical_or(info["stance_mask"], info["last_stance_mask"])
    #     first_contact = (info["feet_air_time"] > 0) * contact_filter
    #     reward = jnp.sum(info["feet_air_time"] * first_contact)
    #     # no reward for zero command
    #     reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
    #     return reward

    # def _reward_feet_clearance(
    #     self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    # ) -> jax.Array:
    #     contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
    #     first_contact = (info["feet_air_dist"] > 0) * contact_filter
    #     reward = jnp.sum(info["feet_air_dist"] * first_contact)
    #     # no reward for zero command
    #     reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
    #     return reward

    def _reward_feet_distance(self, obs: Obs, action: np.ndarray) -> np.ndarray:
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        feet_dist = obs.feet_y_dist
        d_min = np.clip(feet_dist - self.min_feet_y_dist, a_min=-np.inf, a_max=0.0)
        d_max = np.clip(feet_dist - self.max_feet_y_dist, a_min=0.0, a_max=np.inf)
        reward = (np.exp(-np.abs(d_min) * 100) + np.exp(-np.abs(d_max) * 100)) / 2
        return reward

    # def _reward_feet_slip(
    #     self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    # ) -> jax.Array:
    #     feet_speed = pipeline_state.xd.vel[self.feet_link_ids]
    #     feet_speed_square = jnp.square(feet_speed[:, :2])
    #     reward = -jnp.sum(feet_speed_square * info["stance_mask"])
    #     # Penalize large feet velocity for feet that are in contact with the ground.
    #     return reward

    def _reward_stand_still(self, obs: Obs, action: np.ndarray) -> float:
        # Penalize motion at zero commands
        qpos_diff = np.sum(np.abs(obs.motor_pos - self.default_motor_pos))
        reward = -(qpos_diff**2)
        # DISCUSS: reward: -0.06,-> 0
        reward *= np.linalg.norm(self.fixed_command) < self.deadzone
        return reward

    # def _reward_align_ground(self, obs: Obs, action: np.ndarray) -> float:
    #     hip_pitch_joint_pos = jnp.abs(
    #         pipeline_state.q[self.q_start_idx + self.hip_pitch_joint_indices]
    #     )
    #     knee_joint_pos = jnp.abs(
    #         pipeline_state.q[self.q_start_idx + self.knee_joint_indices]
    #     )
    #     ank_pitch_joint_pos = np.abs(
    #         obs.motor_pos[self.ank_pitch_joint_indices]
    #     )
    #     error = hip_pitch_joint_pos + ank_pitch_joint_pos - knee_joint_pos
    #     reward = -np.mean(error**2)
    #     return reward
