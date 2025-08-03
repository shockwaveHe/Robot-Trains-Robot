from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
import mujoco
import numpy as np
import scipy
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.mujoco_control import MotorController
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import (
    butterworth,
    euler2quat,
    exponential_moving_average,
    quat2euler,
    quat_inv,
    quat_mult,
    rotate_vec,
)

dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32  # type: ignore
# Global registry to store env names and their corresponding classes
env_registry: Dict[str, Type["MJXEnv"]] = {}


def get_env_class(env_name: str) -> Type["MJXEnv"]:
    """Returns the environment class associated with the given environment name.

    Args:
        env_name (str): The name of the environment to retrieve.

    Returns:
        Type[MJXEnv]: The class of the specified environment.

    Raises:
        ValueError: If the environment name is not found in the registry.
    """
    if env_name not in env_registry:
        raise ValueError(f"Unknown env: {env_name}")

    return env_registry[env_name]


def get_env_names() -> List[str]:
    """Retrieve a list of environment names.

    Returns:
        List[str]: A list containing the names of all registered environments.
    """
    return list(env_registry.keys())


class MJXEnv(PipelineEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        motion_ref: MotionReference,
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initializes the environment with the specified configuration and robot parameters.

        Args:
            name (str): The name of the environment.
            robot (Robot): The robot instance to be used in the environment.
            cfg (MJXConfig): Configuration settings for the environment and simulation.
            motion_ref (MotionReference): Reference for motion planning and execution.
            fixed_base (bool, optional): Whether the robot has a fixed base. Defaults to False.
            add_noise (bool, optional): Whether to add noise to the simulation. Defaults to True.
            add_domain_rand (bool, optional): Whether to add domain randomization. Defaults to True.
            **kwargs (Any): Additional keyword arguments for environment initialization.
        """
        self.name = name
        self.cfg = cfg
        self.robot = robot
        self.motion_ref = motion_ref
        self.fixed_base = fixed_base
        self.add_noise = add_noise
        self.add_domain_rand = add_domain_rand
        self.use_hang = cfg.hang.use_hang
        # eval in original environment
        if fixed_base:
            xml_path = find_robot_file_path(robot.name, suffix="_fixed_scene.xml")
        elif self.use_hang:
            xml_path = find_robot_file_path(robot.name, suffix="_hang_scene.xml")
            self.hang_positions = [
                "left_front",
                "right_front",
                "left_back",
                "right_back",
            ]
        else:
            xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

        sys = mjcf.load(xml_path)
        sys = sys.tree_replace(
            {
                "opt.timestep": cfg.sim.timestep,
                "opt.solver": cfg.sim.solver,
                "opt.iterations": cfg.sim.iterations,
                "opt.ls_iterations": cfg.sim.ls_iterations,
            }
        )

        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._init_env()
        self._init_reward()

    # Automatic registration of subclasses
    def __init_subclass__(cls, env_name: str = "", **kwargs):
        """Initializes a subclass and optionally registers it in the environment registry.

        Args:
            env_name (str): The name of the environment to register the subclass under. If provided, the subclass is added to the `env_registry` with this name.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        if len(env_name) > 0:
            env_registry[env_name] = cls

    def _init_env(self) -> None:
        """Initializes the environment by setting up various system parameters, colliders, joint indices, motor indices, actuator indices, and action configurations.

        This method configures the environment based on the system and robot specifications, including the number of joints, colliders, and actuators. It identifies and categorizes joint and motor indices for different body parts such as legs, arms, neck, and waist. It also sets up action masks, default actions, and noise scales for the simulation. Additionally, it configures filters and command parameters for controlling the robot's movements and interactions within the environment.
        """
        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv

        if self.use_hang:
            self.hang_motors = 12
            self.hang_dofs = 8
            self.nu -= self.hang_motors
            self.nq -= self.hang_dofs
            self.nv -= self.hang_dofs
        else:
            self.hang_motors = 0
            self.hang_dofs = 0
        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        # colliders
        pair_geom1 = self.sys.pair_geom1
        pair_geom2 = self.sys.pair_geom2
        self.collider_geom_ids = np.unique(np.concatenate([pair_geom1, pair_geom2]))
        self.num_colliders = self.collider_geom_ids.shape[0]
        left_foot_collider_indices: List[int] = []
        right_foot_collider_indices: List[int] = []
        for i, geom_id in enumerate(self.collider_geom_ids):
            geom_name = support.id2name(self.sys, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                continue

            if f"{self.robot.foot_name}_2" in geom_name:
                right_foot_collider_indices.append(i)
            elif f"{self.robot.foot_name}" in geom_name:
                left_foot_collider_indices.append(i)

        self.left_foot_collider_indices = jnp.array(left_foot_collider_indices)
        self.right_foot_collider_indices = jnp.array(right_foot_collider_indices)

        feet_link_mask = jnp.array(
            np.char.find(self.sys.link_names, self.robot.foot_name) >= 0
        )
        self.feet_link_ids = jnp.arange(self.sys.num_links())[feet_link_mask]

        self.contact_force_threshold = self.cfg.action.contact_force_threshold

        # This leads to CPU memory leak
        # self.jit_contact_force = jax.jit(support.contact_force, static_argnums=(2, 3))
        self.jit_contact_force = support.contact_force

        self.joint_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.joint_indices -= 1

        joint_groups = np.array(
            [self.robot.joint_groups[name] for name in self.robot.joint_ordering]
        )
        self.leg_joint_indices = self.joint_indices[joint_groups == "leg"]
        self.arm_joint_indices = self.joint_indices[joint_groups == "arm"]
        self.neck_joint_indices = self.joint_indices[joint_groups == "neck"]
        self.waist_joint_indices = self.joint_indices[joint_groups == "waist"]

        hip_pitch_joint_mask = np.char.find(self.robot.joint_ordering, "hip_pitch") >= 0
        knee_joint_mask = np.char.find(self.robot.joint_ordering, "knee") >= 0
        ank_pitch_joint_mask = np.char.find(self.robot.joint_ordering, "ank_pitch") >= 0

        self.hip_pitch_joint_indices = self.joint_indices[hip_pitch_joint_mask]
        self.knee_joint_indices = self.joint_indices[knee_joint_mask]
        self.ank_pitch_joint_indices = self.joint_indices[ank_pitch_joint_mask]

        self.motor_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.motor_indices -= 1

        motor_groups = np.array(
            [self.robot.joint_groups[name] for name in self.robot.motor_ordering]
        )
        self.leg_motor_indices = self.motor_indices[joint_groups == "leg"]
        self.arm_motor_indices = self.motor_indices[joint_groups == "arm"]
        self.neck_motor_indices = self.motor_indices[joint_groups == "neck"]
        self.waist_motor_indices = self.motor_indices[joint_groups == "waist"]

        hip_motor_mask = np.char.find(self.robot.motor_ordering, "hip") >= 0
        self.hip_motor_indices = self.motor_indices[hip_motor_mask]

        self.actuator_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.leg_actuator_indices = self.actuator_indices[motor_groups == "leg"]
        self.arm_actuator_indices = self.actuator_indices[motor_groups == "arm"]
        self.neck_actuator_indices = self.actuator_indices[motor_groups == "neck"]
        self.waist_actuator_indices = self.actuator_indices[motor_groups == "waist"]

        self.joint_ref_indices = jnp.arange(len(self.robot.joint_ordering))
        self.leg_ref_indices = self.joint_ref_indices[joint_groups == "leg"]
        self.arm_ref_indices = self.joint_ref_indices[joint_groups == "arm"]
        self.neck_ref_indices = self.joint_ref_indices[joint_groups == "neck"]
        self.waist_ref_indices = self.joint_ref_indices[joint_groups == "waist"]

        # default qpos
        self.default_qpos = jnp.array(self.sys.mj_model.keyframe("home").qpos)

        # action
        self.action_parts = self.cfg.action.action_parts
        self.motor_limits = jnp.array(
            [self.robot.joint_limits[name] for name in self.robot.motor_ordering]
        )

        action_mask: List[jax.Array] = []
        default_action: List[jax.Array] = []
        for part_name in self.action_parts:
            if part_name == "leg":
                action_mask.append(self.leg_actuator_indices)
                default_action.append(
                    self.default_qpos[self.q_start_idx + self.leg_motor_indices]
                )
            elif part_name == "waist":
                action_mask.append(self.waist_actuator_indices)
                default_action.append(
                    self.default_qpos[self.q_start_idx + self.waist_motor_indices]
                )
            elif part_name == "arm":
                action_mask.append(self.arm_actuator_indices)
                default_action.append(
                    self.default_qpos[self.q_start_idx + self.arm_motor_indices]
                )
            elif part_name == "neck":
                action_mask.append(self.neck_actuator_indices)
                default_action.append(
                    self.default_qpos[self.q_start_idx + self.neck_motor_indices]
                )

        self.action_mask = jnp.concatenate(action_mask)
        self.num_action = self.action_mask.shape[0]
        self.default_action = jnp.concatenate(default_action)

        self.action_scale = self.cfg.action.action_scale
        self.n_steps_delay = self.cfg.action.n_steps_delay
        self.action_noise_scale = self.cfg.noise.action_noise

        self.controller = MotorController(self.robot)

        # Filter
        self.filter_type = self.cfg.action.filter_type
        self.filter_order = self.cfg.action.filter_order
        # EMA
        self.ema_alpha = float(
            self.cfg.action.filter_cutoff
            / (self.cfg.action.filter_cutoff + 1 / (self.dt * 2 * jnp.pi))
        )
        # Butterworth
        b, a = scipy.signal.butter(
            self.filter_order,
            self.cfg.action.filter_cutoff / (0.5 / self.dt),
            btype="low",
            analog=False,
        )
        self.butter_b_coef = jnp.array(b)[:, None]
        self.butter_a_coef = jnp.array(a)[:, None]

        # commands
        # x vel, y vel, yaw vel, heading
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)
        self.reset_time = self.cfg.commands.reset_time
        self.reset_steps = int(self.reset_time / self.dt)
        self.mean_reversion = self.cfg.commands.mean_reversion
        self.zero_chance = self.cfg.commands.zero_chance
        self.turn_chance = self.cfg.commands.turn_chance
        self.command_obs_indices = jnp.array(self.cfg.commands.command_obs_indices)
        self.command_range = jnp.array(self.cfg.commands.command_range)
        self.deadzone = (
            jnp.array(self.cfg.commands.deadzone)
            if len(self.cfg.commands.deadzone) > 1
            else self.cfg.commands.deadzone[0]
        )
        # observation
        self.ref_start_idx = 7 + 6
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs
        self.obs_scales = self.cfg.obs_scales

        if self.robot.has_gripper:
            self.obs_size += 4
            self.privileged_obs_size += 4

        self.embedding = jnp.empty(0)

        # noise
        self.obs_noise_scale = self.cfg.noise.obs_noise_scale * jnp.concatenate(
            [
                jnp.zeros(
                    self.obs_size
                    - 2 * self.actuator_indices.shape[0]
                    - self.num_action
                    - 6
                ),
                jnp.ones_like(self.actuator_indices) * self.cfg.noise.dof_pos,
                jnp.ones_like(self.actuator_indices) * self.cfg.noise.dof_vel,
                jnp.zeros(self.num_action),
                # jnp.ones(3) * self.cfg.noise.lin_vel,
                jnp.ones(3) * self.cfg.noise.ang_vel,
                jnp.ones(3) * self.cfg.noise.euler,
            ]
        )
        self.tau_limit_scale = self.cfg.noise.tau_limit_scale
        self.backlash_scale = self.cfg.noise.backlash_scale
        self.backlash_activation = self.cfg.noise.backlash_activation

        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_lin_vel = self.cfg.domain_rand.push_lin_vel
        self.push_ang_vel = self.cfg.domain_rand.push_ang_vel

    def _init_reward(self) -> None:
        """Initializes the reward system by filtering and scaling reward components.

        This method processes the reward scales configuration by removing any components with a scale of zero and scaling the remaining components by a time factor. It then prepares a list of reward function names and their corresponding scales, which are stored for later use in reward computation. Additionally, it sets parameters related to health and tracking rewards.
        """
        reward_scale_dict = asdict(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions: List[Callable[..., jax.Array]] = []
        self.reward_scales = jnp.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales = self.reward_scales.at[i].set(scale)

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.tracking_sigma = self.cfg.rewards.tracking_sigma

    @property
    def action_size(self) -> int:  # override default action_size
        """Returns the number of possible actions.

        Overrides the default action size to provide the specific number of actions available.

        Returns:
            int: The number of possible actions.
        """
        return self.num_action

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment state and initializes various components for a new episode.

        This function splits the input random number generator (RNG) into multiple streams for different components, initializes the state information dictionary, and sets up the initial positions, velocities, and commands for the environment. It also applies domain randomization if enabled and prepares observation histories.

        Args:
            rng (jax.Array): The random number generator state for initializing the environment.

        Returns:
            State: The initialized state of the environment, including pipeline state, observations, rewards, and other relevant information.
        """
        rng, rng_torso_yaw, rng_action, rng_command = jax.random.split(rng, 4)

        state_info = {
            "rng": rng,
            "contact_forces": jnp.zeros((self.num_colliders, self.num_colliders, 3)),
            "left_foot_contact_mask": jnp.zeros(len(self.left_foot_collider_indices)),
            "right_foot_contact_mask": jnp.zeros(len(self.right_foot_collider_indices)),
            "feet_air_time": jnp.zeros(2),
            "feet_air_dist": jnp.zeros(2),
            "action_buffer": jnp.zeros((self.n_steps_delay + 1) * self.num_action),
            "last_last_act": jnp.zeros(self.num_action),
            "last_act": jnp.zeros(self.num_action),
            "last_torso_euler": jnp.zeros(3),
            "rewards": {k: 0.0 for k in self.reward_names},
            "push_lin": jnp.zeros(3),
            "push_ang": jnp.zeros(3),
            "done": False,
            "step": 0,
            "hang_force": 0.0,
            "tendon_length": jnp.zeros(4),
        }

        qpos = self.default_qpos.copy()
        qvel = jnp.zeros(self.nv)

        path_pos = jnp.zeros(3)
        path_yaw = jax.random.uniform(rng_torso_yaw, (1,), minval=0, maxval=2 * jnp.pi)
        path_euler = jnp.array([0.0, 0.0, jnp.degrees(path_yaw)[0]])
        path_quat = euler2quat(path_euler)
        lin_vel = jnp.zeros(3)
        ang_vel = jnp.zeros(3)
        motor_pos = qpos[self.q_start_idx + self.motor_indices]
        joint_pos = qpos[self.q_start_idx + self.joint_indices]
        stance_mask = jnp.ones(2)

        state_ref = jnp.concatenate(
            [path_pos, path_quat, lin_vel, ang_vel, motor_pos, joint_pos, stance_mask]
        )
        command = self._sample_command(rng_command)
        state_ref = jnp.asarray(self.motion_ref.get_state_ref(state_ref, 0.0, command))
        qpos = qpos.at[: self.nq].set(
            jnp.asarray(self.motion_ref.get_qpos_ref(state_ref, path_frame=False))
        )
        # jax.debug.print("euler: {}", math.quat_to_euler(torso_quat))
        # jax.debug.print("torso_euler: {}", torso_euler)
        # jax.debug.print("waist_euler: {}", waist_euler)

        qvel = jnp.zeros(self.nv + self.hang_dofs)
        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info["command"] = command
        state_info["command_obs"] = command[self.command_obs_indices]
        state_info["state_ref"] = state_ref
        state_info["stance_mask"] = state_ref[-2:]
        state_info["last_stance_mask"] = state_ref[-2:]
        state_info["phase_signal"] = self.motion_ref.get_phase_signal(0.0)
        state_info["feet_height_init"] = pipeline_state.x.pos[self.feet_link_ids, 2]
        last_motor_target = pipeline_state.qpos[self.q_start_idx + self.motor_indices]
        last_action_target = last_motor_target[self.action_mask]
        state_info["last_action_target"] = last_action_target
        state_info["butter_past_inputs"] = jnp.tile(
            last_action_target, (self.filter_order, 1)
        )
        state_info["butter_past_outputs"] = jnp.tile(
            last_action_target, (self.filter_order, 1)
        )
        state_info["tendon_length"] = pipeline_state.ten_length[-4:]
        state_info["default_action"] = self.default_action.copy() + jax.random.uniform(
            rng_action,
            (self.num_action,),
            minval=-self.action_noise_scale,
            maxval=self.action_noise_scale,
        )

        obs_history = jnp.concatenate(
            [jnp.zeros(self.num_obs_history * self.obs_size), self.embedding], axis=-1
        )
        privileged_obs_history = jnp.concatenate(
            [
                jnp.zeros(self.num_privileged_obs_history * self.privileged_obs_size),
                self.embedding,
            ],
            axis=-1,
        )
        obs, privileged_obs = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history,
        )
        reward, done, zero = jnp.zeros(3)

        metrics: Dict[str, Any] = {}
        for k in self.reward_names:
            metrics[k] = zero

        metrics["episode_num"] = 0.0
        metrics["hang_force"] = state_info["hang_force"]

        # jax.debug.print("current running time: {time}", time = time.time() - self.episode_start_time)

        return State(
            pipeline_state, obs, privileged_obs, reward, done, metrics, state_info
        )

    def pipeline_step(self, state: State, action: jax.Array) -> base.State:
        """Executes a pipeline step by applying a control action to the system state.

        This function iteratively applies a control action to the system's state over a specified number of frames. It uses a controller to compute control signals based on the current state and action, and updates the pipeline state accordingly.

        Args:
            state (State): The current state of the system, containing information required for control computations.
            action (jax.Array): The control action to be applied to the system.

        Returns:
            base.State: The updated state of the system after applying the control action over the specified number of frames.
        """
        rng, tau_limit_rng = jax.random.split(state.info["rng"], 2)
        if self.use_hang:
            progress = jnp.minimum(
                state.info.get("episode_num", 0) / self.cfg.hang.hang_force_decay_episodes,
                1.0,
            )
            # hang_force = self.cfg.hang.init_hang_force - progress * (
            #     self.cfg.hang.init_hang_force - self.cfg.hang.final_hang_force
            # )
            phase = progress * jnp.pi
            # Compute the cosine value, which transitions from 1 to 0
            cosine_value = 0.5 * (1 + jnp.cos(phase))
            # Calculate the hang force using the cosine schedule
            hang_force = self.cfg.hang.final_hang_force + cosine_value * (
                self.cfg.hang.init_hang_force - self.cfg.hang.final_hang_force
            )
            state.info["hang_force"] = hang_force

        def f(pipeline_state, _):
            ctrl = self.controller.step(
                pipeline_state.q[self.q_start_idx + self.motor_indices],
                pipeline_state.qd[self.qd_start_idx + self.motor_indices],
                action,
                self.tau_limit_scale * jax.random.normal(tau_limit_rng, action.shape),
            )

            # concatenate hang_motors dim to ctrl
            if self.use_hang > 0:
                ctrl = jnp.concatenate([ctrl, jnp.zeros(self.hang_motors)])
                torso_xy = pipeline_state.q[:2]
                position_offset = [
                    [0.022, 0.045],
                    [-0.022, 0.045],
                    [0.022, -0.045],
                    [-0.022, -0.045],
                ]
                tendon_length = pipeline_state.ten_length[-4:]

                for i, pos in enumerate(position_offset):
                    ctrl = ctrl.at[self.nu + 3 * i].set(
                        -state.info["hang_force"]
                        - self.cfg.hang.tendon_kp
                        * (tendon_length[i] - jnp.mean(tendon_length))
                    )
                    ctrl = ctrl.at[self.nu + 3 * i + 1].set(pos[0] + torso_xy[0])
                    ctrl = ctrl.at[self.nu + 3 * i + 2].set(pos[1] + torso_xy[1])

            return (
                self._pipeline.step(self.sys, pipeline_state, ctrl, self._debug),
                None,
            )

        return jax.lax.scan(f, state.pipeline_state, (), self._n_frames)[0]

    def step(self, state: State, action: jax.Array) -> State:
        """Advances the simulation by one time step, updating the state based on the given action.

        This function updates the state of the simulation by processing the given action, applying filters, and incorporating domain randomization if enabled. It computes the motor targets, updates the pipeline state, and checks for termination conditions. Additionally, it calculates rewards and updates various state information, including contact forces, stance masks, and command resampling.

        Args:
            state (State): The current state of the simulation, containing information about the system's dynamics and metadata.
            action (jax.Array): The action to be applied at this time step, influencing the system's behavior.

        Returns:
            State: The updated state after applying the action and advancing the simulation by one step.
        """
        rng, cmd_rng, push_lin_rng, push_ang_rng = jax.random.split(
            state.info["rng"], 4
        )

        time_curr = state.info["step"] * self.dt
        state_ref = self.motion_ref.get_state_ref(
            state.info["state_ref"], time_curr, state.info["command"]
        )
        state.info["state_ref"] = state_ref
        state.info["phase_signal"] = self.motion_ref.get_phase_signal(time_curr)
        state.info["action_buffer"] = (
            jnp.roll(state.info["action_buffer"], self.num_action)
            .at[: self.num_action]
            .set(action)
        )

        delayed_action = state.info["action_buffer"][-self.num_action :]
        action_target = (
            state.info["default_action"] + self.action_scale * delayed_action
        )

        if self.filter_type == "ema":
            action_target = exponential_moving_average(
                self.ema_alpha, action_target, state.info["last_action_target"]
            )
        elif self.filter_type == "butter":
            (
                action_target,
                state.info["butter_past_inputs"],
                state.info["butter_past_outputs"],
            ) = butterworth(
                self.butter_b_coef,
                self.butter_a_coef,
                action_target,
                state.info["butter_past_inputs"],
                state.info["butter_past_outputs"],
            )

        assert isinstance(action_target, jax.Array)
        state.info["last_action_target"] = action_target.copy()

        motor_target = (
            jnp.asarray(state_ref[self.ref_start_idx : self.ref_start_idx + self.nu])
            .at[self.action_mask]
            .set(action_target)
        )

        motor_target = jnp.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        if self.add_domain_rand:
            linear_vel = jax.random.uniform(
                push_lin_rng, (3,), minval=-self.push_lin_vel, maxval=self.push_lin_vel
            )
            angular_vel = jax.random.uniform(
                push_ang_rng, (3,), minval=-self.push_ang_vel, maxval=self.push_ang_vel
            )

            # Apply the push only at certain intervals
            push_mask = jnp.mod(state.info["step"], self.push_interval) == 0
            linear_vel *= push_mask
            angular_vel *= push_mask

            # Apply the sampled velocities to the system
            qvel = state.pipeline_state.qd
            qvel = qvel.at[:3].add(linear_vel)  # Update linear velocity
            qvel = qvel.at[3:6].add(angular_vel)  # Update angular velocity

            state = state.tree_replace({"pipeline_state.qd": qvel})
            state.info["push_lin"] = linear_vel
            state.info["push_ang"] = angular_vel

        # jax.debug.breakpoint()
        pipeline_state = self.pipeline_step(state, motor_target)
        state.info["tendon_length"] = pipeline_state.ten_length[-4:]

        # jax.debug.print(
        #     "qfrc: {}",
        #     pipeline_state.qfrc_actuator[self.qd_start_idx + self.leg_motor_indices],
        # )
        # jax.debug.print("stance_mask: {}", state.info["stance_mask"])
        # jax.debug.print("feet_air_time: {}", state.info["feet_air_time"])
        # jax.debug.print("feet_air_dist: {}", state.info["feet_air_dist"])

        if not self.fixed_base:
            contact_forces, left_foot_contact_mask, right_foot_contact_mask = (
                self._get_contact_forces(pipeline_state)
            )
            stance_mask = jnp.array(
                [jnp.any(left_foot_contact_mask), jnp.any(right_foot_contact_mask)]
            ).astype(dtype)

            state.info["contact_forces"] = contact_forces
            state.info["left_foot_contact_mask"] = left_foot_contact_mask
            state.info["right_foot_contact_mask"] = right_foot_contact_mask
            state.info["stance_mask"] = stance_mask

        torso_height = pipeline_state.x.pos[0, 2]
        done = jnp.logical_or(
            torso_height < self.healthy_z_range[0],
            torso_height > self.healthy_z_range[1],
        )
        state.info["done"] = done

        obs, privileged_obs = self._get_obs(
            pipeline_state, state.info, state.obs, state.privileged_obs
        )

        torso_euler = quat2euler(pipeline_state.x.rot[0])
        torso_euler_delta = torso_euler - state.info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        torso_euler = state.info["last_torso_euler"] + torso_euler_delta

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = sum(reward_dict.values()) * self.dt
        # reward = jnp.clip(reward, 0.0)

        if not self.fixed_base:
            state.info["last_stance_mask"] = stance_mask.copy()
            state.info["feet_air_time"] += self.dt
            state.info["feet_air_time"] *= 1.0 - stance_mask

            feet_z_delta = (
                pipeline_state.x.pos[self.feet_link_ids, 2]
                - state.info["feet_height_init"]
            )
            state.info["feet_air_dist"] += feet_z_delta
            state.info["feet_air_dist"] *= 1.0 - stance_mask

        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = delayed_action.copy()
        state.info["last_torso_euler"] = torso_euler
        state.info["rewards"] = reward_dict
        state.info["rng"] = rng
        state.info["step"] += 1

        # jax.debug.print("step: {}", state.info["step"])

        state.info["command"] = jax.lax.cond(
            state.info["step"] % self.resample_steps
            == 0,  # TODO: set resample_steps to a large value, no last_command
            lambda: self._sample_command(cmd_rng, state.info["command"]),
            lambda: state.info["command"],
        )
        state.info["command_obs"] = state.info["command"][self.command_obs_indices]

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self.reset_steps), 0, state.info["step"]
        )
        state.metrics.update(reward_dict)
        state.metrics.update(
            {
                "episode_num": state.info.get("episode_num", 0.0),
                "hang_force": state.info["hang_force"],
            }
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            privileged_obs=privileged_obs,
            reward=reward,
            done=done.astype(dtype),
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        raise NotImplementedError

    def _sample_command_uniform(
        self, rng: jax.Array, command_range: jax.Array
    ) -> jax.Array:
        """Generates a uniformly distributed random sample within specified command ranges.

        Args:
            rng (jax.Array): A JAX random number generator array.
            command_range (jax.Array): A 2D array where each row specifies the minimum and maximum values for sampling.

        Returns:
            jax.Array: An array of uniformly distributed random samples, one for each range specified in `command_range`.
        """
        return jax.random.uniform(
            rng,
            (command_range.shape[0],),
            minval=command_range[:, 0],
            maxval=command_range[:, 1],
        )

    def _sample_command_normal(
        self, rng: jax.Array, command_range: jax.Array
    ) -> jax.Array:
        """Samples a command from a normal distribution and clips it to a specified range.

        Args:
            rng (jax.Array): Random number generator array for sampling.
            command_range (jax.Array): Array specifying the range for each command dimension,
                where each row is [min, max].

        Returns:
            jax.Array: An array of sampled commands, clipped to the specified range.
        """
        return jnp.clip(
            jax.random.normal(rng, (command_range.shape[0],))
            * command_range[:, 1]
            / 3.0,
            command_range[:, 0],
            command_range[:, 1],
        )

    def _sample_command_normal_reversion(
        self, rng: jax.Array, command_range: jax.Array, last_command: jax.Array
    ) -> jax.Array:
        """Generates a sample command using normal distribution with mean reversion.

        This function samples a command from a normal distribution, applies mean reversion to the last command, and clips the result within specified command ranges.

        Args:
            rng (jax.Array): Random number generator array for sampling.
            command_range (jax.Array): Array specifying the min and max range for each command dimension.
            last_command (jax.Array): The last command array to apply mean reversion.

        Returns:
            jax.Array: A new command array sampled and adjusted according to the specified parameters.
        """
        return jnp.clip(
            jax.random.normal(rng, (command_range.shape[0],))
            * command_range[:, 1]
            / 3.0
            - self.mean_reversion * last_command,
            command_range[:, 0],
            command_range[:, 1],
        )

    def _get_contact_forces(self, data: mjx.Data):
        """Compute contact forces between colliders and determine foot contact masks.

        This function calculates the contact forces between colliders based on the provided
        simulation data. It also determines whether the left and right foot colliders are in
        contact with the ground by comparing the contact forces against a predefined threshold.

        Args:
            data (mjx.Data): The simulation data containing contact information.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: A tuple containing:
                - A 3D array of shape (num_colliders, num_colliders, 3) representing the global
                  contact forces between colliders.
                - A 1D array indicating whether each left foot collider is in contact.
                - A 1D array indicating whether each right foot collider is in contact.
        """
        # Extract geom1 and geom2 directly
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2

        def get_body_index(geom_id: jax.Array) -> jax.Array:
            return jnp.argmax(self.collider_geom_ids == geom_id)

        # Vectorized computation of body indices for geom1 and geom2
        body_indices_1 = jax.vmap(get_body_index)(geom1)
        body_indices_2 = jax.vmap(get_body_index)(geom2)

        contact_forces_global = jnp.zeros((self.num_colliders, self.num_colliders, 3))
        for i in range(data.ncon):
            contact_force = self.jit_contact_force(self.sys, data, i, True)[:3]
            # Update the contact forces for both body_indices_1 and body_indices_2
            # Add instead of set to accumulate forces from multiple contacts
            contact_forces_global = contact_forces_global.at[
                body_indices_1[i], body_indices_2[i]
            ].add(contact_force)
            contact_forces_global = contact_forces_global.at[
                body_indices_2[i], body_indices_1[i]
            ].add(contact_force)

        left_foot_contact_mask = (
            contact_forces_global[0, self.left_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(dtype)
        right_foot_contact_mask = (
            contact_forces_global[0, self.right_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(dtype)

        return contact_forces_global, left_foot_contact_mask, right_foot_contact_mask

    def _get_obs(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        obs_history: jax.Array,
        privileged_obs_history: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generates and returns the current and privileged observations for the system.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing position, velocity, and other dynamics information.
            info (dict[str, Any]): A dictionary containing additional information such as random number generator state, reference states, and other auxiliary data.
            obs_history (jax.Array): An array storing the history of observations for the system.
            privileged_obs_history (jax.Array): An array storing the history of privileged observations for the system.

        Returns:
            Tuple[jax.Array, jax.Array]: A tuple containing the updated observation and privileged observation arrays.
        """
        rng, obs_rng = jax.random.split(info["rng"], 2)

        motor_pos = pipeline_state.q[self.q_start_idx + self.motor_indices]
        motor_pos_delta = (
            motor_pos - self.default_qpos[self.q_start_idx + self.motor_indices]
        )
        motor_backlash = self.backlash_scale * jnp.tanh(
            pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
            / self.backlash_activation
        )

        motor_vel = pipeline_state.qd[self.qd_start_idx + self.motor_indices]

        motor_pos_error = (
            motor_pos
            - info["state_ref"][self.ref_start_idx : self.ref_start_idx + self.nu]
        )

        torso_quat = pipeline_state.x.rot[0]
        torso_lin_vel = rotate_vec(pipeline_state.xd.vel[0], quat_inv(torso_quat))
        torso_ang_vel = rotate_vec(pipeline_state.xd.ang[0], quat_inv(torso_quat))

        torso_euler = quat2euler(torso_quat)
        torso_euler_delta = torso_euler - info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        torso_euler = info["last_torso_euler"] + torso_euler_delta

        obs = jnp.concatenate(
            [
                info["phase_signal"],
                info["command_obs"],
                motor_pos_delta * self.obs_scales.dof_pos + motor_backlash,
                motor_vel * self.obs_scales.dof_vel,
                info["last_act"],
                # motor_pos_error,
                # torso_lin_vel * self.obs_scales.lin_vel,
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_euler * self.obs_scales.euler,
            ]
        )
        privileged_obs = jnp.concatenate(
            [
                info["phase_signal"],
                info["command_obs"],
                motor_pos_delta * self.obs_scales.dof_pos,
                motor_vel * self.obs_scales.dof_vel,
                info["last_act"],
                motor_pos_error,
                torso_lin_vel * self.obs_scales.lin_vel,
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_euler * self.obs_scales.euler,
                info["stance_mask"],
                info["state_ref"][-2:],
                info["push_lin"],
                info["push_ang"],
            ]
        )

        if self.add_noise:
            obs += self.obs_noise_scale * jax.random.normal(obs_rng, obs.shape)

        # jax.debug.breakpoint()
        # obs = jnp.clip(obs, -100.0, 100.0)

        # stack observations through time
        obs_history = obs_history.at[: self.num_obs_history * self.obs_size].set(
            jnp.roll(obs_history[: self.num_obs_history * self.obs_size], obs.size)
        )
        obs_history = obs_history.at[: obs.size].set(obs)

        privileged_obs_history = privileged_obs_history.at[
            : self.num_privileged_obs_history * self.privileged_obs_size
        ].set(
            jnp.roll(
                privileged_obs_history[
                    : self.num_privileged_obs_history * self.privileged_obs_size
                ],
                privileged_obs.size,
            )
        )
        privileged_obs_history = privileged_obs_history.at[: privileged_obs.size].set(
            privileged_obs
        )

        return obs_history, privileged_obs_history

    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Computes a dictionary of rewards based on the current pipeline state, additional information, and the action taken.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): Additional information that may be required for reward computation.
            action (jax.Array): The action taken, which influences the reward calculation.

        Returns:
            Dict[str, jax.Array]: A dictionary where keys are reward names and values are the computed rewards as JAX arrays.
        """
        # Create an array of indices to map over
        indices = jnp.arange(len(self.reward_names))
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(
            lambda i: jax.lax.switch(
                i,
                self.reward_functions,
                pipeline_state,
                info,
                action,
            )
            * self.reward_scales[i],
            indices,
        )

        reward_dict: Dict[str, jax.Array] = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict

    def _reward_torso_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the position of the torso.

        This function computes a reward by comparing the current position of the torso
        to a reference position. The reward is calculated using a Gaussian function
        that penalizes deviations from the reference position.

        Args:
            pipeline_state (base.State): The current state of the system, containing
                positional information.
            info (dict[str, Any]): A dictionary containing reference state information,
                specifically the reference position of the torso.
            action (jax.Array): The action taken, though not used in this reward calculation.

        Returns:
            jax.Array: The computed reward based on the deviation of the torso's position
            from the reference position.
        """
        torso_pos = pipeline_state.x.pos[0][:2]  # Assuming [:2] extracts xy components
        torso_pos_ref = info["state_ref"][:2]
        error = jnp.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = jnp.exp(-200.0 * error**2)
        return reward.astype(dtype)

    def _reward_torso_quat(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the alignment of the torso's quaternion orientation with a reference orientation.

        Args:
            pipeline_state (base.State): The current state of the system, containing the rotation of the torso.
            info (dict[str, Any]): A dictionary containing reference state information, including the reference quaternion and waist joint positions.
            action (jax.Array): The action taken, though not used in this function.

        Returns:
            jax.Array: A reward value computed from the quaternion angle difference between the current and reference torso orientations.
        """
        torso_quat = pipeline_state.x.rot[0]
        path_quat_ref = info["state_ref"][3:7]

        waist_joint_pos = info["state_ref"][
            self.ref_start_idx + self.nu + self.waist_ref_indices
        ]
        waist_euler = jnp.array([waist_joint_pos[0], 0.0, waist_joint_pos[1]])
        torso_quat_ref = quat_mult(
            path_quat_ref, quat_inv(euler2quat(jnp.degrees(waist_euler)))
        )

        # Quaternion dot product (cosine of the half-angle)
        dot_product = jnp.sum(torso_quat * torso_quat_ref, axis=-1)
        # Ensure the dot product is within the valid range
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        # Quaternion angle difference
        angle_diff = 2.0 * jnp.arccos(jnp.abs(dot_product))
        reward = jnp.exp(-20.0 * (angle_diff**2))
        return reward.astype(dtype)

    def _reward_lin_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the linear velocity in the XY plane.

        Args:
            pipeline_state (base.State): The current state of the system, including position and velocity.
            info (dict[str, Any]): Additional information, including reference state data.
            action (jax.Array): The action taken by the agent.

        Returns:
            jax.Array: The computed reward value, which is a function of the error between the current and reference linear velocities in the XY plane.
        """
        lin_vel_local = rotate_vec(
            pipeline_state.xd.vel[0], quat_inv(pipeline_state.x.rot[0])
        )
        lin_vel_xy = lin_vel_local[:2]
        lin_vel_xy_ref = info["state_ref"][7:9]
        error = jnp.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.tracking_sigma * error**2)
        return reward.astype(dtype)

    def _reward_lin_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculate the reward based on the vertical component of linear velocity.

        This function computes a reward that measures how closely the vertical component
        of the linear velocity of an object matches a reference value. The reward is
        calculated using a Gaussian function of the error between the actual and
        reference vertical velocities.

        Args:
            pipeline_state (base.State): The current state of the system, containing
                position and velocity information.
            info (dict[str, Any]): A dictionary containing reference state information,
                specifically the reference vertical velocity.
            action (jax.Array): The action taken, not used in this calculation.

        Returns:
            jax.Array: The computed reward based on the vertical velocity tracking error.
        """
        lin_vel_local = rotate_vec(
            pipeline_state.xd.vel[0], quat_inv(pipeline_state.x.rot[0])
        )
        lin_vel_z = lin_vel_local[2]
        lin_vel_z_ref = info["state_ref"][9]
        error = lin_vel_z - lin_vel_z_ref
        reward = jnp.exp(-self.tracking_sigma * error**2)
        return reward.astype(dtype)

    def _reward_ang_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the angular velocity in the XY plane.

        This function computes the reward by comparing the current angular velocity in the XY plane to a reference value. The reward is calculated using a Gaussian function of the error between the current and reference angular velocities.

        Args:
            pipeline_state (base.State): The current state of the system, containing rotational and angular velocity information.
            info (dict[str, Any]): A dictionary containing reference state information, specifically the target angular velocity in the XY plane.
            action (jax.Array): The action taken, though not directly used in this function.

        Returns:
            jax.Array: The computed reward based on the angular velocity tracking error.
        """
        ang_vel_local = rotate_vec(
            pipeline_state.xd.ang[0], quat_inv(pipeline_state.x.rot[0])
        )
        ang_vel_xy = ang_vel_local[:2]
        ang_vel_xy_ref = info["state_ref"][10:12]
        error = jnp.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)
        return reward.astype(dtype)

    def _reward_ang_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculate the reward based on the z-component of angular velocity.

        This function computes a reward that measures how closely the z-component of the
        angular velocity of a system matches a reference value. The reward is calculated
        using a Gaussian function of the error between the actual and reference angular
        velocities.

        Args:
            pipeline_state (base.State): The current state of the system, including
                angular velocities and orientations.
            info (dict[str, Any]): A dictionary containing reference states, including
                the reference angular velocity.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed reward based on the angular velocity error.
        """
        ang_vel_local = rotate_vec(
            pipeline_state.xd.ang[0], quat_inv(pipeline_state.x.rot[0])
        )
        ang_vel_z = ang_vel_local[2]
        ang_vel_z_ref = info["state_ref"][12]
        error = ang_vel_z - ang_vel_z_ref
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)
        return reward.astype(dtype)

    def _reward_feet_contact(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the contact of feet with the ground.

        This function computes the reward by comparing the stance mask from the
        `info` dictionary with the reference state, specifically the last two
        elements of the `state_ref` array. The reward is the sum of matches
        between these two arrays, indicating the number of feet in contact with
        the ground as expected.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about the
                current state, including the 'stance_mask' and 'state_ref'.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.numpy.ndarray: The computed reward as a float32 value.
        """
        reward = jnp.sum(info["stance_mask"] == info["state_ref"][-2:]).astype(
            jnp.float32
        )
        return reward

    def _reward_leg_motor_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the position error of leg motors.

        This function computes the reward by evaluating the squared error between the current leg motor positions and the reference positions. The reward is the negative mean of these squared errors, encouraging the motor positions to match the reference.

        Args:
            pipeline_state (base.State): The current state of the system, containing the positions of all motors.
            info (dict[str, Any]): A dictionary containing reference state information, including the desired motor positions.
            action (jax.Array): The action taken, though not used in this reward calculation.

        Returns:
            jax.Array: The calculated reward as a negative mean of the squared position errors.
        """
        motor_pos = pipeline_state.q[self.q_start_idx + self.leg_motor_indices]
        motor_pos_ref = info["state_ref"][self.ref_start_idx + self.leg_ref_indices]
        error = motor_pos - motor_pos_ref
        reward = -jnp.mean(error**2)
        return reward.astype(dtype)

    def _reward_arm_motor_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the position error of the arm motor.

        This function computes the reward by evaluating the mean squared error between the current motor positions and the reference motor positions. The reward is negative, indicating that smaller errors result in higher rewards.

        Args:
            pipeline_state (base.State): The current state of the system, containing the motor positions.
            info (dict[str, Any]): A dictionary containing reference state information, including the target motor positions.
            action (jax.Array): The action taken, though not used in this function.

        Returns:
            jax.Array: The calculated reward as a negative mean squared error of the motor position differences.
        """
        motor_pos = pipeline_state.q[self.q_start_idx + self.arm_motor_indices]
        motor_pos_ref = info["state_ref"][self.ref_start_idx + self.arm_ref_indices]
        error = motor_pos - motor_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_neck_motor_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the neck motor positions by comparing the current motor positions to reference positions.

        Args:
            pipeline_state (base.State): The current state of the pipeline containing motor positions.
            info (dict[str, Any]): A dictionary containing reference state information.
            action (jax.Array): The action taken, not used in this function.

        Returns:
            jax.Array: The calculated reward as a negative mean squared error between current and reference neck motor positions.
        """
        motor_pos = pipeline_state.q[self.q_start_idx + self.neck_motor_indices]
        motor_pos_ref = info["state_ref"][self.ref_start_idx + self.neck_ref_indices]
        error = motor_pos - motor_pos_ref
        reward = -jnp.mean(error**2)
        return reward.astype(dtype)

    def _reward_waist_motor_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the position error of the waist motor.

        This function computes the reward by evaluating the mean squared error between the current waist motor positions and the reference positions. A lower error results in a higher reward.

        Args:
            pipeline_state (base.State): The current state of the system, containing the positions of all motors.
            info (dict[str, Any]): A dictionary containing reference state information, including the desired motor positions.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: The calculated reward as a JAX array, where a lower position error results in a higher reward.
        """
        motor_pos = pipeline_state.q[self.q_start_idx + self.waist_motor_indices]
        motor_pos_ref = info["state_ref"][self.ref_start_idx + self.waist_ref_indices]
        error = motor_pos - motor_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_collision(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a negative reward based on collision forces in the environment.

        This function computes a penalty for collisions by evaluating the contact forces
        between objects, excluding the floor. If the force exceeds a threshold, it is
        considered a collision, and a negative reward is accumulated for each collision.

        Args:
            pipeline_state (base.State): The current state of the simulation pipeline.
            info (dict[str, Any]): A dictionary containing information about the current
                simulation step, including contact forces.
            action (jax.Array): The action taken by the agent, not used in this function.

        Returns:
            float: A negative reward representing the penalty for collisions.
        """
        collision_forces = jnp.linalg.norm(
            info["contact_forces"][1:, 1:],  # exclude the floor
            axis=-1,
        )
        collision_contact = collision_forces > 0.1
        reward = -jnp.sum(collision_contact.astype(dtype))
        return reward.astype(dtype)

    def _reward_motor_torque(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on motor torque.

        This function computes a reward by evaluating the squared error of the motor torque
        and returning its negative mean. The reward is designed to penalize high torque values.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing actuator forces.
            info (dict[str, Any]): Additional information, not used in this function.
            action (jax.Array): The action taken, not used in this function.

        Returns:
            jax.Array: The calculated reward as a negative mean of the squared torque error.
        """
        torque = pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
        error = jnp.square(torque)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_energy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the energy-based reward for a given pipeline state and action.

        This function computes the reward based on the energy consumption of the actuators
        in the system. It calculates the energy as the product of torque and motor velocity,
        then computes the error as the square of the energy. The reward is the negative mean
        of this error, encouraging actions that minimize energy consumption.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing
                information about actuator forces and velocities.
            info (dict[str, Any]): Additional information that might be used for reward
                calculation (not used in this function).
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: The calculated reward as a JAX array, representing the negative mean
            of the squared energy error.
        """
        torque = pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
        motor_vel = pipeline_state.qvel[self.qd_start_idx + self.motor_indices]
        energy = torque * motor_vel
        error = jnp.square(energy)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_leg_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the rate of change of leg actions.

        This function computes a reward by evaluating the squared difference between the current and last leg actions, averaged over all leg actuators. The reward is negative, encouraging minimal change in leg actions.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in this function.
            info (dict[str, Any]): A dictionary containing the last action taken, with key "last_act".
            action (jax.Array): The current action array, from which leg actions are extracted.

        Returns:
            jax.Array: A scalar reward representing the negative mean squared error of leg action changes.
        """
        leg_action = action[self.leg_actuator_indices]
        last_leg_action = info["last_act"][self.leg_actuator_indices]
        error = jnp.square(leg_action - last_leg_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_leg_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the acceleration of leg actions.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about previous actions.
            action (jax.Array): The current action array.

        Returns:
            jax.Array: The calculated reward as a negative mean squared error of the leg action acceleration.
        """
        leg_action = action[self.leg_actuator_indices]
        last_leg_action = info["last_act"][self.leg_actuator_indices]
        last_last_leg_action = info["last_last_act"][self.leg_actuator_indices]
        error = jnp.square(leg_action - 2 * last_leg_action + last_last_leg_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_arm_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the rate of change of arm actions.

        This function computes the reward by evaluating the squared error between the current and last arm actions, and returns the negative mean of these errors. The reward is designed to penalize large changes in arm actions, encouraging smoother transitions.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in this function.
            info (dict[str, Any]): A dictionary containing additional information, specifically the last action taken.
            action (jax.Array): The current action array, from which the arm actions are extracted.

        Returns:
            jax.Array: The calculated reward as a negative mean of the squared differences between current and last arm actions.
        """
        arm_action = action[self.arm_actuator_indices]
        last_arm_action = info["last_act"][self.arm_actuator_indices]
        error = jnp.square(arm_action - last_arm_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_arm_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculate the reward based on the acceleration of arm actions.

        This function computes the reward by evaluating the acceleration of arm actions
        using the difference between the current, last, and second-to-last actions. The
        reward is the negative mean of the squared error of this acceleration, promoting
        smooth transitions in arm movements.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in
                the computation.
            info (dict[str, Any]): A dictionary containing information about previous
                actions, specifically 'last_act' and 'last_last_act' for arm actuators.
            action (jax.Array): The current action array from which arm actuator actions
                are extracted.

        Returns:
            jax.Array: The calculated reward as a negative mean of the squared error of
            the arm action acceleration.
        """
        arm_action = action[self.arm_actuator_indices]
        last_arm_action = info["last_act"][self.arm_actuator_indices]
        last_last_arm_action = info["last_last_act"][self.arm_actuator_indices]
        error = jnp.square(arm_action - 2 * last_arm_action + last_last_arm_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_neck_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the rate of change of neck actuator actions.

        This function computes a reward by evaluating the squared difference between
        the current and previous actions of neck actuators, and then returns the
        negative mean of these squared differences. The reward is designed to
        penalize large changes in neck actuator actions, encouraging smoother
        movements.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing additional information,
                including the last actions taken.
            action (jax.Array): An array representing the current actions for all
                actuators.

        Returns:
            jax.Array: A scalar reward value representing the penalty for the rate
            of change in neck actuator actions.
        """
        neck_action = action[self.neck_actuator_indices]
        last_neck_action = info["last_act"][self.neck_actuator_indices]
        error = jnp.square(neck_action - last_neck_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_neck_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward for neck actuator actions based on acceleration error.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about previous actions, specifically the last and second-to-last neck actuator actions.
            action (jax.Array): The current action array containing actions for all actuators.

        Returns:
            jax.Array: The calculated reward as a negative mean squared error of the neck actuator acceleration.
        """
        neck_action = action[self.neck_actuator_indices]
        last_neck_action = info["last_act"][self.neck_actuator_indices]
        last_last_neck_action = info["last_last_act"][self.neck_actuator_indices]
        error = jnp.square(neck_action - 2 * last_neck_action + last_last_neck_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_waist_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the rate of change of waist actuator actions.

        This function computes a reward by evaluating the squared difference between the current and last actions of the waist actuators. The reward is the negative mean of these squared differences, encouraging minimal change in actuator actions.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in this function.
            info (dict[str, Any]): A dictionary containing additional information, including the last actions under the key "last_act".
            action (jax.Array): The current action array, from which waist actuator actions are extracted.

        Returns:
            jax.Array: A scalar reward representing the negative mean squared error of the waist actuator actions.
        """
        waist_action = action[self.waist_actuator_indices]
        last_waist_action = info["last_act"][self.waist_actuator_indices]
        error = jnp.square(waist_action - last_waist_action)
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_waist_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the acceleration of the waist actuator actions.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in this function.
            info (dict[str, Any]): A dictionary containing information about previous actions, specifically the last and second-to-last waist actuator actions.
            action (jax.Array): The current action array from which the waist actuator actions are extracted.

        Returns:
            jax.Array: A scalar reward value calculated as the negative mean squared error of the waist actuator acceleration.
        """
        waist_action = action[self.waist_actuator_indices]
        last_waist_action = info["last_act"][self.waist_actuator_indices]
        last_last_waist_action = info["last_last_act"][self.waist_actuator_indices]
        error = jnp.square(
            waist_action - 2 * last_waist_action + last_last_waist_action
        )
        reward = -jnp.mean(error)
        return reward.astype(dtype)

    def _reward_survival(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a survival reward based on the pipeline state and action taken.

        The reward is negative if the episode is marked as done before reaching the
        specified number of reset steps, encouraging survival until the reset threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing episode information, including
                whether the episode is done and the current step count.
            action (jax.Array): The action taken at the current step.

        Returns:
            jax.Array: A float32 array representing the survival reward.
        """
        return -(info["done"] & (info["step"] < self.reset_steps)).astype(jnp.float32)

    def _reward_tendon_len_eq(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tendon length equilibrium"""
        tendon_length = info["tendon_length"]
        error = tendon_length - jnp.mean(tendon_length)
        reward = -jnp.mean(error**2)
        return reward.astype(dtype)
