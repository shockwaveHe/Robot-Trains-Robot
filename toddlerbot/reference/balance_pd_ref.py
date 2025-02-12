from typing import Tuple

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class BalancePDReference(MotionReference):
    """Class for generating a balance reference using a PD controller for the toddlerbot robot."""

    def __init__(
        self,
        robot: Robot,
        dt: float,
        arm_playback_speed: float = 1.0,
    ):
        """Initializes the balance PD controller with specified parameters.

        Args:
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the controller.
            arm_playback_speed (float, optional): The speed factor for arm playback. Defaults to 1.0.
        """
        super().__init__("balance_pd", "perceptual", robot, dt)

        self.arm_playback_speed = arm_playback_speed
        if self.arm_playback_speed > 0.0:
            self.arm_time_ref /= arm_playback_speed

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Calculates linear and angular velocities based on the given command.

        Args:
            command (ArrayType): An array where the sixth element is used to set the linear velocity along the z-axis.

        Returns:
            Tuple[ArrayType, ArrayType]: A tuple containing the linear velocity array and the angular velocity array.
        """
        lin_vel = np.array([0.0, 0.0, command[5]], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        """Calculates the reference state for a robot based on the current state, time, and command inputs.

        This function integrates the path state, updates joint and motor positions for the neck, arms, waist, and legs, and applies a Proportional-Derivative (PD) controller on the Center of Mass (CoM) position to compute the reference state.

        Args:
            state_curr (ArrayType): The current state of the robot.
            time_curr (float | ArrayType): The current time or time array.
            command (ArrayType): The command input array.

        Returns:
            ArrayType: The reference state array, including path state, motor positions, joint positions, and stance mask.
        """
        path_state = self.integrate_path_state(state_curr, command)

        joint_pos_curr = state_curr[13 + self.robot.nu : 13 + 2 * self.robot.nu]

        joint_pos = self.default_joint_pos.copy()
        motor_pos = self.default_motor_pos.copy()
        # neck yaw, neck pitch, arm, waist roll, waist yaw
        neck_joint_pos = np.clip(
            joint_pos_curr[self.neck_joint_indices] + self.dt * command[:2],
            self.neck_joint_limits[0],
            self.neck_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.neck_motor_indices, self.neck_ik(neck_joint_pos)
        )

        if self.arm_playback_speed > 0:
            command_ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
            time_ref = self.arm_time_ref[command_ref_idx] + time_curr
            ref_idx = np.minimum(
                np.searchsorted(self.arm_time_ref, time_ref, side="right") - 1,
                self.arm_ref_size - 2,
            )
            # Linearly interpolate between p_start and p_end
            arm_joint_pos_start = self.arm_joint_pos_ref[ref_idx]
            arm_joint_pos_end = self.arm_joint_pos_ref[ref_idx + 1]
            arm_duration = self.arm_time_ref[ref_idx + 1] - self.arm_time_ref[ref_idx]
            arm_joint_pos = arm_joint_pos_start + (
                arm_joint_pos_end - arm_joint_pos_start
            ) * ((time_ref - self.arm_time_ref[ref_idx]) / arm_duration)
        else:
            arm_joint_pos = joint_pos_curr[self.arm_joint_indices]

        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.arm_motor_indices, self.arm_ik(arm_joint_pos)
        )

        waist_joint_pos = np.clip(
            joint_pos_curr[self.waist_joint_indices] + self.dt * command[3:5],
            self.waist_joint_limits[0],
            self.waist_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.waist_motor_indices, self.waist_ik(waist_joint_pos)
        )
        com_z_target = np.clip(
            state_curr[2] + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_joint_pos = self.com_ik(com_z_target)
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.leg_motor_indices, self.leg_ik(leg_joint_pos)
        )

        state_ref = np.concatenate((path_state, motor_pos, joint_pos))
        qpos = self.get_qpos_ref(state_ref)
        data = self.forward(qpos)

        com_pos = np.array(data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = self.desired_com[:2] - com_pos[:2]
        com_ctrl = self.com_kp * com_pos_error

        # print(
        #     f"com_pos: {com_pos}, com_pos_error: {com_pos_error}, com_ctrl: {com_ctrl}"
        # )

        leg_joint_pos = self.com_ik(com_z_target, com_ctrl[0], com_ctrl[1])
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.leg_motor_indices, self.leg_ik(leg_joint_pos)
        )

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))
