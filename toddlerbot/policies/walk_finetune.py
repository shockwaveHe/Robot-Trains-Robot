from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from toddlerbot.sim import Obs

from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick

class WalkFinetunePolicy(MJXFinetunePolicy, policy_name="walk_finetune"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = get_env_config("walk")
        self.cycle_time = env_cfg.action.cycle_time
        self.command_discount_factor = np.array([0.5, 1.0, 0.75], dtype=np.float32)

        super().__init__(
            name, robot, init_motor_pos, ckpt, joystick, fixed_command, env_cfg
        )
        self.torso_roll_range = env_cfg.rewards.torso_roll_range
        self.torso_pitch_range = env_cfg.rewards.torso_pitch_range

        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = env_cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = env_cfg.rewards.max_feet_y_dist

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
        control_inputs, motor_target = super().step(obs, is_real)

        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            all_zeros = all(np.all(command == 0) for command in last_commands)
            self.is_standing = all_zeros and abs(self.phase_signal[0]) > 1 - 1e-6
        else:
            self.is_standing = False

        return control_inputs, motor_target

    def _reward_torso_roll(self, obs: Obs, action: np.ndarray) -> float:
        """Reward for torso pitch"""
        torso_roll = obs.euler[0]

        roll_min = np.clip(torso_roll - self.torso_roll_range[0], a_min=-np.inf, a_max=0.0)
        roll_max = np.clip(torso_roll - self.torso_roll_range[1], a_min=0.0, a_max=np.inf)
        reward = (
            np.exp(-np.abs(roll_min) * 100) + np.exp(-np.abs(roll_max) * 100)
        ) / 2
        return reward

    def _reward_torso_pitch(self, obs: Obs, action: np.ndarray) -> float:

        """Reward for torso pitch"""
        torso_pitch = obs.euler[1]

        pitch_min = np.clip(torso_pitch - self.torso_pitch_range[0], a_min=-np.inf, a_max=0.0)
        pitch_max = np.clip(torso_pitch - self.torso_pitch_range[1], a_min=0.0, a_max=np.inf)
        reward = (
            np.exp(-np.abs(pitch_min) * 100) + np.exp(-np.abs(pitch_max) * 100)
        ) / 2
        return reward

    # TODO: Implement the foot rewards
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

    # def _reward_feet_distance(
    #     self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    # ):
    #     # Calculates the reward based on the distance between the feet.
    #     # Penalize feet get close to each other or too far away on the y axis
    #     feet_vec = math.rotate(
    #         pipeline_state.x.pos[self.feet_link_ids[0]]
    #         - pipeline_state.x.pos[self.feet_link_ids[1]],
    #         math.quat_inv(pipeline_state.x.rot[0]),
    #     )
    #     feet_dist = jnp.abs(feet_vec[1])
    #     d_min = jnp.clip(feet_dist - self.min_feet_y_dist, max=0.0)
    #     d_max = jnp.clip(feet_dist - self.max_feet_y_dist, min=0.0)
    #     reward = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2
    #     return reward

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
        qpos_diff = np.sum(
            np.abs(obs.motor_pos - self.default_motor_pos)
        )
        reward = -(qpos_diff**2)
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
