from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.finetuning.finetune_config import get_finetune_config
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from scipy.spatial.transform import Rotation
from toddlerbot.utils.math_utils import euler2quat


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
    ):
        if env_cfg is None:
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

        super().__init__(
            name,
            robot,
            init_motor_pos,
            ckpts,
            ip,
            eval_mode,
            joystick,
            fixed_command,
            env_cfg,
            finetune_cfg,
            exp_folder=exp_folder,
            is_real=is_real,
        )

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

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target, obs = super().step(obs, is_real)

        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            all_zeros = all(np.all(command == 0) for command in last_commands)
            self.is_standing = all_zeros and abs(self.phase_signal[0]) > 1 - 1e-6
        else:
            self.is_standing = False
        self.last_torso_yaw = obs.euler[2]
        return control_inputs, motor_target, obs

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

    # TODO: Implement the foot rewards
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

    # TODO: Implement the reward for aligning the ground?
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
