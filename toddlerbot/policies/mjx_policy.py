import functools
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import torch
import yaml
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

from toddlerbot.locomotion.actor_critic import ActorCritic
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action  # , soft_clamp

# from toddlerbot.utils.misc_utils import profile


def load_runner_config(train_cfg: PPOConfig):
    with open("toddlerbot/locomotion/rsl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        for key, value in config["runner"].items():
            config[key] = value

        del config["runner"]

        config["policy"]["actor_hidden_dims"] = train_cfg.policy_hidden_layer_sizes
        config["policy"]["critic_hidden_dims"] = train_cfg.value_hidden_layer_sizes
        config["max_iterations"] = train_cfg.num_timesteps // (
            train_cfg.num_envs * train_cfg.unroll_length
        )
        config["save_interval"] = config["max_iterations"] // train_cfg.num_evals
        config["num_steps_per_env"] = train_cfg.unroll_length
        config["algorithm"]["gamma"] = train_cfg.discounting
        config["algorithm"]["num_learning_epochs"] = train_cfg.num_updates_per_batch
        config["algorithm"]["learning_rate"] = train_cfg.learning_rate
        config["algorithm"]["entropy_coef"] = train_cfg.entropy_cost
        config["algorithm"]["clip_param"] = train_cfg.clipping_epsilon
        config["algorithm"]["num_mini_batches"] = train_cfg.num_minibatches
        config["seed"] = train_cfg.seed

    return config


class MJXPolicy(BasePolicy, policy_name="mjx"):
    """Policy for controlling the robot using the MJX model."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        cfg: Optional[MJXConfig] = None,
        use_torch: bool = False,
        exp_folder: Optional[str] = "",
        need_warmup: Optional[bool] = True,
    ):
        """Initializes the class with configuration and state parameters for controlling a robot.

        Args:
            name (str): The name of the robot controller.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            ckpt (str): Path to the checkpoint file for loading model parameters.
            joystick (Optional[Joystick]): Joystick instance for manual control, if available.
            fixed_command (Optional[npt.NDArray[np.float32]]): Fixed command array, if any.
            cfg (Optional[MJXConfig]): Configuration object containing control parameters.
            use_torch (bool): Flag to indicate whether to use PyTorch for inference.
        Raises:
            AssertionError: If `cfg` is not provided.
        """
        super().__init__(name, robot, init_motor_pos, exp_folder=exp_folder)

        assert cfg is not None, "cfg is required in the subclass!"

        self.ckpt = ckpt
        self.cfg = cfg
        self.use_torch = use_torch

        self.command_obs_indices = cfg.commands.command_obs_indices
        self.commmand_range = np.array(cfg.commands.command_range, dtype=np.float32)
        self.num_commands = len(self.commmand_range)

        if fixed_command is None:
            self.fixed_command = np.zeros(self.num_commands, dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.obs_history_size = cfg.obs.frame_stack * cfg.obs.num_single_obs
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)

        self.obs_scales = cfg.obs_scales  # Assume all the envs have the same scales
        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        self.action_scale = cfg.action.action_scale
        self.n_steps_delay = cfg.action.n_steps_delay
        self.action_parts = cfg.action.action_parts
        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        motor_groups = np.array(
            [self.robot.joint_groups[name] for name in self.robot.motor_ordering]
        )
        actuator_indices = np.arange(len(self.robot.motor_ordering))
        action_mask: List[npt.NDArray[np.float32]] = []
        default_action: List[npt.NDArray[np.float32]] = []
        for part_name in self.action_parts:
            if part_name == "leg":
                action_mask.append(actuator_indices[motor_groups == "leg"])
                default_action.append(self.default_motor_pos[motor_groups == "leg"])
            elif part_name == "waist":
                action_mask.append(actuator_indices[motor_groups == "waist"])
                default_action.append(self.default_motor_pos[motor_groups == "waist"])
            elif part_name == "arm":
                action_mask.append(actuator_indices[motor_groups == "arm"])
                default_action.append(self.default_motor_pos[motor_groups == "arm"])
            elif part_name == "neck":
                action_mask.append(actuator_indices[motor_groups == "neck"])
                default_action.append(self.default_motor_pos[motor_groups == "neck"])

        self.action_mask = np.concatenate(action_mask)
        self.num_action = self.action_mask.shape[0]
        self.default_action = np.concatenate(default_action)

        self.inference_fn = None

        # Filter
        self.filter_type = self.cfg.action.filter_type
        self.filter_order = self.cfg.action.filter_order
        # EMA
        self.ema_alpha = float(
            self.cfg.action.filter_cutoff
            / (self.cfg.action.filter_cutoff + 1 / (self.control_dt * 2 * np.pi))
        )

        if need_warmup:
            self.warmup_result: Dict[str, Any] = {}
            self.warmup_event = threading.Event()

            self.warmup_thread = threading.Thread(
                target=self.warmup,
                args=(self.warmup_result, self.warmup_event),
                daemon=True,
            )
            self.warmup_thread.start()

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        self.control_inputs: Dict[str, float] = {}
        self.is_prepared = False

        self.reset()

    def warmup(self, result_container, event):
        """Initializes and loads a policy for the agent, preparing it for inference.

        This method sets up the necessary configurations and loads a pre-trained policy
        from a specified path. It compiles the policy for efficient execution and stores
        the compiled inference function and random number generator in a shared result
        container. It also signals the completion of the setup process.

        Args:
            result_container (dict): A shared container to store the compiled inference
                function and random number generator.
            event (threading.Event): An event object used to signal the completion of
                the warmup process.
        """
        try:
            if "walk" in self.name:
                policy_name = "walk"
            else:
                policy_name = self.name

            if len(self.ckpt) > 0:
                run_name = f"{self.robot.name}_{policy_name}_ppo_{self.ckpt}"

                if self.use_torch:
                    policy_path = os.path.join("results", run_name, "model_best.pt")
                else:
                    policy_path = os.path.join("results", run_name, "best_policy")
                    if not os.path.exists(policy_path):
                        policy_path = os.path.join("results", run_name, "policy")
            else:
                policy_path = os.path.join(
                    "toddlerbot", "policies", "checkpoints", f"{policy_name}_policy"
                )

            print(f"Loading policy from {policy_path}")
            train_cfg = PPOConfig()

            if self.use_torch:
                runner_config = load_runner_config(train_cfg)
                alg_class = eval(runner_config["policy"].pop("class_name"))
                actor_critic: ActorCritic = alg_class(
                    self.cfg.obs.num_single_obs * self.cfg.obs.frame_stack,
                    self.cfg.obs.num_single_privileged_obs * self.cfg.obs.c_frame_stack,
                    self.num_action,
                    **runner_config["policy"],
                )
                state_dict = torch.load(policy_path, map_location="cpu")
                actor_critic.load_state_dict(state_dict["model_state_dict"])
                actor_critic.eval()

                def policy(obs: torch.Tensor):
                    with torch.no_grad():
                        action = actor_critic.act_inference(obs)
                        # action = soft_clamp(action, -1.0, 1.0)
                        return action

            else:
                make_networks_factory = functools.partial(
                    ppo_networks.make_ppo_networks,
                    policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
                    value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
                )

                ppo_network = make_networks_factory(
                    self.cfg.obs.num_single_obs,
                    self.cfg.obs.num_single_privileged_obs,
                    self.num_action,
                )
                make_policy = ppo_networks.make_inference_fn(ppo_network)

                params = model.load_params(policy_path)
                inference_fn = jax.jit(make_policy(params, deterministic=True))
                rng = jax.random.PRNGKey(0)

                def policy(obs: jax.Array):
                    return inference_fn(obs, rng)[0]

            # Store results in the shared container
            result_container["inference_fn"] = policy

        finally:
            # Signal that the thread is done
            event.set()

    def reset(self, obs: Obs = None):
        """Resets the internal state of the policy to its initial configuration.

        This method clears the observation history, phase signal, command list, and action buffer. It also sets the standing state to True and initializes the last action and current step counter to zero.
        """
        print(f"Resetting the {self.name} policy...")
        self.obs_history = np.zeros(self.obs_history_size, dtype=np.float32)
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.is_standing = True
        self.command_list = []
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.step_curr = 0

    def get_phase_signal(self, time_curr: float) -> npt.NDArray[np.float32]:
        """Get the phase signal at the current time.

        Args:
            time_curr (float): The current time for which the phase signal is requested.

        Returns:
            npt.NDArray[np.float32]: An array containing the phase signal as a float32 value.
        """
        return np.zeros(1, dtype=np.float32)

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        """Returns a fixed command as a NumPy array.

        Args:
            control_inputs (Dict[str, float]): A dictionary of control inputs, where keys are input names and values are their respective float values.

        Returns:
            npt.NDArray[np.float32]: A fixed command represented as a NumPy array of float32 values.
        """
        return self.fixed_command

    # @profile()
    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes a single step in the control loop, updating the system's state and generating motor target positions.

        Args:
            obs (Obs): The current observation containing motor positions, velocities, and other sensor data.
            is_real (bool, optional): Indicates if the system is operating in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs and the target motor positions.
        """
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
            action_np = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action_np, None

        if self.inference_fn is None:
            self.warmup_event.wait()  # Block until the event is set
            self.inference_fn = self.warmup_result["inference_fn"]

        assert self.inference_fn is not None, "inference_fn is not set!"

        time_curr = self.step_curr * self.control_dt

        control_inputs: Dict[str, float] = {}
        if len(self.control_inputs) > 0:
            control_inputs = self.control_inputs
        elif self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()

        if len(control_inputs) == 0:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        self.phase_signal = self.get_phase_signal(time_curr)
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        motor_vel = obs.motor_vel

        if self.robot.has_gripper:
            motor_pos_delta = motor_pos_delta[:-2]
            motor_vel = motor_vel[:-2]

        obs_arr = np.concatenate(
            [
                self.phase_signal,
                command[self.command_obs_indices],
                motor_pos_delta * self.obs_scales.dof_pos,
                motor_vel * self.obs_scales.dof_vel,
                self.last_action,
                # motor_pos_error,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales.ang_vel,
                obs.euler * self.obs_scales.euler,
            ]
        )

        self.obs_history = np.roll(self.obs_history, obs_arr.size)
        self.obs_history[: obs_arr.size] = obs_arr

        # if np.any(np.abs(obs.euler) > np.pi):
        #     euler_delta = obs.euler - ((obs.euler + np.pi) % (2 * np.pi) - np.pi)
        #     obs_history_reshape = self.obs_history.reshape(-1, obs_arr.size).copy()
        #     obs_history_reshape[:, -3:] -= euler_delta
        #     obs_history = obs_history_reshape.flatten()
        # else:
        # obs_history = self.obs_history

        if self.use_torch:
            action = self.inference_fn(torch.tensor(self.obs_history))
            action_np = action.numpy().copy()
        else:
            action = self.inference_fn(jnp.asarray(self.obs_history))
            action_np = np.asarray(action, dtype=np.float32).copy()

        if is_real:
            delayed_action = action_np
        else:
            self.action_buffer = np.roll(self.action_buffer, action_np.size)
            self.action_buffer[: action_np.size] = action_np
            delayed_action = self.action_buffer[-self.num_action :]

        action_target = self.default_action + self.action_scale * delayed_action

        # motor_target = self.state_ref[13 : 13 + self.robot.nu].copy()
        motor_target = self.default_motor_pos.copy()
        motor_target[self.action_mask] = action_target

        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.command_list.append(command)
        self.last_action = delayed_action
        self.step_curr += 1

        return control_inputs, motor_target, None
