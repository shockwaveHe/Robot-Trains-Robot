import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import joblib
import mujoco
import numpy
from mujoco import mjx

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import euler2quat, quat_inv, quat_mult, rotate_vec


class MotionReference(ABC):
    """Abstract class for generating motion references for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        motion_type: str,
        robot: Robot,
        dt: float,
        com_kp: List[float] = [1.0, 1.0],
    ):
        """Initializes the motion controller for a robot with specified parameters.

        Args:
            name (str): The name of the motion controller.
            motion_type (str): The type of motion to be controlled (e.g., 'walking', 'running').
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the control loop.
            com_kp (List[float], optional): The proportional gain for the center of mass control. Defaults to [1.0, 1.0].
        """
        self.name = name
        self.motion_type = motion_type
        self.robot = robot
        self.dt = dt
        self.com_kp = np.array(com_kp, dtype=np.float32)
        self.use_jax = os.environ.get("USE_JAX", "false") == "true"

        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        indices = np.arange(robot.nu)
        motor_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.motor_ordering]
        )
        joint_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        self._setup_neck()
        self._setup_arm()
        self._setup_waist()
        self._setup_leg()
        self._setup_mjx()

    def _get_gear_ratios(self, motor_names: List[str]) -> ArrayType:
        """Calculates the gear ratios for specified motors.

        Args:
            motor_names (List[str]): A list of motor names for which to calculate gear ratios.

        Returns:
            ArrayType: An array of gear ratios corresponding to the provided motor names.
        """
        gear_ratios = np.ones(len(motor_names), dtype=np.float32)
        for i, motor_name in enumerate(motor_names):
            motor_config = self.robot.config["joints"][motor_name]
            if motor_config["transmission"] in ["gear", "rack_and_pinion"]:
                gear_ratios = inplace_update(
                    gear_ratios, i, -motor_config["gear_ratio"]
                )
        return gear_ratios

    def _setup_neck(self):
        """Initializes the neck configuration for the robot.

        This method sets up the necessary parameters for controlling the neck
        of the robot, including motor names, gear ratios, and joint limits.
        It identifies the indices of the neck motors, calculates the gear
        ratios, and retrieves the joint limits for neck yaw and pitch.
        """
        neck_motor_names = [
            self.robot.motor_ordering[i] for i in self.neck_motor_indices
        ]
        self.neck_pitch_idx = self.robot.joint_ordering.index("neck_pitch")
        self.neck_gear_ratio = self._get_gear_ratios(neck_motor_names)
        self.neck_joint_limits = np.array(
            [
                self.robot.joint_limits["neck_yaw_driven"],
                self.robot.joint_limits["neck_pitch"],
            ],
            dtype=np.float32,
        ).T

    def _setup_arm(self):
        """Initializes the arm setup by configuring motor names, calculating gear ratios, and loading reference datasets for arm motion. This includes setting up time and joint position references based on a preloaded dataset, and ensuring compatibility with the expected number of motors."""
        arm_motor_names = [self.robot.motor_ordering[i] for i in self.arm_motor_indices]
        self.arm_gear_ratio = self._get_gear_ratios(arm_motor_names)

        # Load the balance dataset
        data_path = os.path.join("motion", "arm_pose_dataset.lz4")
        data_dict = joblib.load(data_path)
        time_arr = data_dict["time"]
        motor_pos_arr = data_dict["motor_pos"]
        self.arm_time_ref = np.array(time_arr - time_arr[0], dtype=np.float32)
        if motor_pos_arr.shape[1] < len(arm_motor_names):
            gripper_padding = np.zeros((motor_pos_arr.shape[0], 2), dtype=np.float32)
            motor_pos_arr = np.concatenate([motor_pos_arr, gripper_padding], axis=1)

        self.arm_joint_pos_ref = np.array(
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in motor_pos_arr[:, : len(arm_motor_names)]
            ],
            dtype=np.float32,
        )
        self.arm_ref_size = len(self.arm_time_ref)

    def _setup_waist(self):
        """Initializes the waist configuration and joint limits for the robot.

        Sets up the waist coefficients and joint limits using the robot's configuration
        and joint limit data. The coefficients and limits are stored as numpy arrays
        with a float32 data type.
        """
        self.waist_coef = np.array(
            [
                self.robot.config["general"]["offsets"]["waist_roll_coef"],
                self.robot.config["general"]["offsets"]["waist_yaw_coef"],
            ],
            dtype=np.float32,
        )
        self.waist_joint_limits = np.array(
            [
                self.robot.joint_limits["waist_roll"],
                self.robot.joint_limits["waist_yaw"],
            ],
            dtype=np.float32,
        ).T

    def _setup_leg(self):
        """Initializes the leg configuration by setting up motor names, gear ratios, and joint indices.

        This method retrieves the motor names for the leg based on predefined indices, calculates the gear ratios for these motors, and determines the indices of key joints in the robot's joint ordering. These joints include the left and right knee, hip pitch, and hip roll.
        """
        leg_motor_names = [self.robot.motor_ordering[i] for i in self.leg_motor_indices]
        self.leg_gear_ratio = self._get_gear_ratios(leg_motor_names)
        self.left_knee_idx = self.robot.joint_ordering.index("left_knee")
        self.left_hip_pitch_idx = self.robot.joint_ordering.index("left_hip_pitch")
        self.left_hip_roll_idx = self.robot.joint_ordering.index("left_hip_roll")
        self.right_knee_idx = self.robot.joint_ordering.index("right_knee")
        self.right_hip_pitch_idx = self.robot.joint_ordering.index("right_hip_pitch")
        self.right_hip_roll_idx = self.robot.joint_ordering.index("right_hip_roll")

    def _setup_mjx(self, com_z_lower_limit_offset: float = 0.01):
        """Initializes and configures the MuJoCo model for the robot, setting up joint indices, site IDs, and kinematic parameters.

        Args:
            com_z_lower_limit_offset (float): Offset to apply to the lower limit of the center of mass (COM) in the z-direction. Default is 0.01.
        """
        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        # self.renderer = mujoco.Renderer(model)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.mj_passive_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.passive_joint_names
            ]
        )
        # Account for the free joint
        self.mj_joint_indices -= 1
        self.mj_motor_indices -= 1
        self.mj_passive_indices -= 1

        self.left_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_center"
        )
        self.passive_joint_indices = np.repeat(
            np.array([self.neck_pitch_idx, self.left_knee_idx, self.right_knee_idx]), 4
        )
        self.passive_joint_signs = np.repeat(np.array([-1, 1, 1], dtype=np.float32), 4)

        if "gripper" in self.robot.name:
            self.passive_joint_indices = np.concatenate(
                [
                    self.passive_joint_indices,
                    np.array(
                        [
                            self.robot.joint_ordering.index("left_gripper_pinion"),
                            self.robot.joint_ordering.index("right_gripper_pinion"),
                        ]
                    ),
                ]
            )
            self.passive_joint_signs = np.concatenate(
                [self.passive_joint_signs, np.array([1, 1], dtype=np.float32)]
            )

        hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "2xc430")
        hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip_yaw_link")
        knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_calf_link")
        ank_pitch_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_ank_pitch_link"
        )
        ank_roll_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ank_roll_link"
        )

        if self.use_jax:
            self.model = mjx.put_model(model)

            def forward(qpos):
                data = mjx.make_data(self.model)
                data = data.replace(qpos=qpos)
                return mjx.forward(self.model, data)

        else:
            self.model = model

            def forward(qpos):
                data = mujoco.MjData(self.model)
                data.qpos = qpos
                mujoco.mj_forward(self.model, data)
                return data

        self.forward = forward

        data = self.forward(self.default_qpos)
        self.left_foot_center = np.asarray(data.site_xpos[self.left_foot_site_id])
        self.right_foot_center = np.asarray(data.site_xpos[self.right_foot_site_id])
        self.torso_pos_init = np.asarray(data.qpos[:3])
        self.desired_com = (self.left_foot_center + self.right_foot_center) / 2.0
        # self.desired_com = np.array(data.subtree_com[0], dtype=np.float32)

        self.knee_default = self.default_joint_pos[self.left_knee_idx]
        self.knee_max = np.max(
            np.abs(np.array(self.robot.joint_limits["left_knee"], dtype=np.float32))
        )

        hip_to_knee_default = np.asarray(
            data.xpos[hip_pitch_id] - data.xpos[knee_id],
            dtype=np.float32,
        )
        self.hip_to_knee_len = np.sqrt(
            hip_to_knee_default[0] ** 2 + hip_to_knee_default[2] ** 2
        )
        knee_to_ank_default = np.asarray(
            data.xpos[knee_id] - data.xpos[ank_pitch_id],
            dtype=np.float32,
        )
        self.knee_to_ank_len = np.sqrt(
            knee_to_ank_default[0] ** 2 + knee_to_ank_default[2] ** 2
        )

        hip_to_ank_pitch_default = np.asarray(
            data.xpos[hip_pitch_id] - data.xpos[ank_pitch_id],
            dtype=np.float32,
        )
        hip_to_ank_roll_default = np.asarray(
            data.xpos[hip_roll_id] - data.xpos[ank_roll_id],
            dtype=np.float32,
        )
        self.hip_to_ank_pitch_default = inplace_update(hip_to_ank_pitch_default, 1, 0.0)
        self.hip_to_ank_roll_default = inplace_update(hip_to_ank_roll_default, 0, 0.0)

        self.com_z_limits = np.array(
            [self.com_fk(self.knee_max)[2] + com_z_lower_limit_offset, 0.0],
            dtype=np.float32,
        )

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        """Calculate the phase signal at a given time.

        Args:
            time_curr (float | ArrayType): The current time or an array of time values.

        Returns:
            ArrayType: An array containing the phase signal, initialized to zeros.
        """
        return np.zeros(1, dtype=np.float32)

    @abstractmethod
    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        pass

    def integrate_path_state(
        self, state_curr: ArrayType, command: ArrayType
    ) -> ArrayType:
        """Integrates the current state of a path with a given command to compute the next state.

        This function calculates the new position and orientation of an object by integrating
        its current state with a command that specifies linear and angular velocities. The
        orientation is updated using quaternion multiplication to apply roll, pitch, and yaw
        rotations.

        Args:
            state_curr (ArrayType): The current state of the object, including position,
                orientation (as a quaternion), linear velocity, and angular velocity.
            command (ArrayType): The command input specifying desired linear and angular velocities.

        Returns:
            ArrayType: The updated state of the object, including new position, orientation,
            linear velocity, and angular velocity.
        """
        lin_vel, ang_vel = self.get_vel(command)

        # Compute the angle of rotation for each axis
        theta_roll = ang_vel[0] * self.dt / 2.0
        theta_pitch = ang_vel[1] * self.dt / 2.0
        theta_yaw = ang_vel[2] * self.dt / 2.0

        # Compute the quaternion for each rotational axis
        roll_quat = np.array([np.cos(theta_roll), np.sin(theta_roll), 0.0, 0.0])
        pitch_quat = np.array([np.cos(theta_pitch), 0.0, np.sin(theta_pitch), 0.0])
        yaw_quat = np.array([np.cos(theta_yaw), 0.0, 0.0, np.sin(theta_yaw)])

        # Normalize each quaternion
        roll_quat /= np.linalg.norm(roll_quat)
        pitch_quat /= np.linalg.norm(pitch_quat)
        yaw_quat /= np.linalg.norm(yaw_quat)

        # Combine the quaternions to get the full rotation (roll * pitch * yaw)
        full_quat = quat_mult(quat_mult(roll_quat, pitch_quat), yaw_quat)

        # Update the current quaternion by applying the new rotation
        path_quat = quat_mult(state_curr[3:7], full_quat)
        path_quat /= np.linalg.norm(path_quat)

        # Update position
        path_pos = state_curr[:3] + rotate_vec(lin_vel, path_quat) * self.dt

        return np.concatenate([path_pos, path_quat, lin_vel, ang_vel])

    @abstractmethod
    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        pass

    def neck_fk(self, neck_motor_pos: ArrayType) -> ArrayType:
        """Calculates the neck joint positions from the neck motor positions.

        Args:
            neck_motor_pos (ArrayType): The positions of the neck motors.

        Returns:
            ArrayType: The calculated positions of the neck joints.
        """
        neck_joint_pos = neck_motor_pos * self.neck_gear_ratio
        return neck_joint_pos

    def neck_ik(self, neck_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for the neck based on the joint positions.

        Args:
            neck_joint_pos (ArrayType): The positions of the neck joints.

        Returns:
            ArrayType: The calculated motor positions for the neck.
        """
        neck_motor_pos = neck_joint_pos / self.neck_gear_ratio
        return neck_motor_pos

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        """Calculates the forward kinematics for an arm by converting motor positions to joint positions.

        Args:
            arm_motor_pos (ArrayType): An array of motor positions for the arm.

        Returns:
            ArrayType: An array of joint positions corresponding to the given motor positions.
        """
        arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for an arm based on joint positions and gear ratio.

        Args:
            arm_joint_pos (ArrayType): The joint positions of the arm.

        Returns:
            ArrayType: The calculated motor positions for the arm.
        """
        arm_motor_pos = arm_joint_pos / self.arm_gear_ratio
        return arm_motor_pos

    def waist_fk(self, waist_motor_pos: ArrayType) -> ArrayType:
        """Calculates the forward kinematics for the waist joint based on motor positions.

        Args:
            waist_motor_pos (ArrayType): An array containing the positions of the waist motors.

        Returns:
            ArrayType: An array containing the calculated waist roll and yaw angles.
        """
        waist_roll = self.waist_coef[0] * (-waist_motor_pos[0] + waist_motor_pos[1])
        waist_yaw = self.waist_coef[1] * (waist_motor_pos[0] + waist_motor_pos[1])
        return np.array([waist_roll, waist_yaw], dtype=np.float32)

    def waist_ik(self, waist_joint_pos: ArrayType) -> ArrayType:
        """Calculates the inverse kinematics for the waist joint actuators.

        Args:
            waist_joint_pos (ArrayType): The position of the waist joint, represented as an array.

        Returns:
            ArrayType: An array containing the calculated positions for the two waist joint actuators.
        """
        waist_roll, waist_yaw = waist_joint_pos / self.waist_coef
        waist_act_1 = (-waist_roll + waist_yaw) / 2
        waist_act_2 = (waist_roll + waist_yaw) / 2
        return np.array([waist_act_1, waist_act_2], dtype=np.float32)

    def leg_fk(self, leg_motor_pos: ArrayType) -> ArrayType:
        """Converts leg motor positions to leg joint positions using the gear ratio.

        Args:
            leg_motor_pos (ArrayType): An array of motor positions for the leg.

        Returns:
            ArrayType: An array of joint positions for the leg, calculated by applying the gear ratio to the motor positions.
        """
        leg_joint_pos = leg_motor_pos * self.leg_gear_ratio
        return leg_joint_pos

    def leg_ik(self, leg_joint_pos: ArrayType) -> ArrayType:
        """Calculates the motor positions for a leg based on joint positions and gear ratio.

        Args:
            leg_joint_pos (ArrayType): The joint positions of the leg.

        Returns:
            ArrayType: The calculated motor positions for the leg.
        """
        leg_motor_pos = leg_joint_pos / self.leg_gear_ratio
        return leg_motor_pos

    def com_fk(
        self,
        knee_angle: float | ArrayType,
        hip_pitch_angle: Optional[float | ArrayType] = None,
        hip_roll_angle: Optional[float | ArrayType] = None,
    ) -> ArrayType:
        """Calculates the center of mass (CoM) position of a leg segment based on joint angles.

        Args:
            knee_angle (float | ArrayType): The angle of the knee joint in radians.
            hip_pitch_angle (Optional[float | ArrayType]): The angle of the hip pitch joint in radians. Defaults to None.
            hip_roll_angle (Optional[float | ArrayType]): The angle of the hip roll joint in radians. Defaults to None.

        Returns:
            ArrayType: A 3D vector representing the CoM position in Cartesian coordinates.
        """
        hip_to_ank = np.sqrt(
            self.hip_to_knee_len**2
            + self.knee_to_ank_len**2
            - 2
            * self.hip_to_knee_len
            * self.knee_to_ank_len
            * np.cos(np.pi - knee_angle)
        )

        if hip_pitch_angle is None:
            alpha = 0.0
        else:
            alpha = (
                np.arcsin(self.knee_to_ank_len / hip_to_ank * np.sin(knee_angle))
                + hip_pitch_angle
            )

        if hip_roll_angle is None:
            hip_roll_angle = 0.0

        com_x = hip_to_ank * np.sin(alpha) * np.cos(hip_roll_angle)
        com_y = hip_to_ank * np.cos(alpha) * np.sin(hip_roll_angle)
        com_z = (
            hip_to_ank * np.cos(alpha) * np.cos(hip_roll_angle)
            - self.hip_to_ank_pitch_default[2]
        )
        return np.array([com_x, com_y, com_z], dtype=np.float32)

    def com_ik(
        self,
        com_z: float | ArrayType,
        com_x: Optional[float | ArrayType] = None,
        com_y: Optional[float | ArrayType] = None,
    ) -> ArrayType:
        """Calculates the inverse kinematics for the center of mass (COM) of a bipedal robot leg.

        This function computes the joint angles required to position the robot's leg such that the center of mass is at the specified coordinates. It uses the lengths of the leg segments and default positions to determine the necessary joint angles.

        Args:
            com_z (float or ArrayType): The z-coordinate of the center of mass.
            com_x (Optional[float or ArrayType]): The x-coordinate of the center of mass. Defaults to 0.0 if not provided.
            com_y (Optional[float or ArrayType]): The y-coordinate of the center of mass. Defaults to 0.0 if not provided.

        Returns:
            ArrayType: An array of joint angles for the leg, including hip pitch, hip roll, knee, and ankle pitch.
        """
        if com_x is None:
            com_x = 0.0
        if com_y is None:
            com_y = 0.0

        hip_to_ank_pitch_vec = self.hip_to_ank_pitch_default + np.array(
            [com_x, 0, com_z], dtype=np.float32
        )
        hip_to_ank_roll_vec = self.hip_to_ank_roll_default + np.array(
            [0, com_y, com_z], dtype=np.float32
        )

        knee_cos = (
            self.hip_to_knee_len**2
            + self.knee_to_ank_len**2
            - np.linalg.norm(hip_to_ank_pitch_vec) ** 2
        ) / (2 * self.hip_to_knee_len * self.knee_to_ank_len)
        knee_cos = np.clip(knee_cos, -1.0, 1.0)
        knee = np.abs(np.pi - np.arccos(knee_cos))

        alpha = np.arctan2(
            self.hip_to_ank_pitch_default[0] + com_x,
            self.hip_to_ank_pitch_default[2] + com_z,
        )
        ank_pitch = (
            np.arctan2(
                np.sin(knee), np.cos(knee) + self.knee_to_ank_len / self.hip_to_knee_len
            )
            + alpha
        )
        hip_pitch = knee - ank_pitch

        hip_roll_cos = np.dot(hip_to_ank_roll_vec, np.array([0, 0, 1])) / (
            np.linalg.norm(hip_to_ank_roll_vec)
        )
        hip_roll_cos = np.clip(hip_roll_cos, -1.0, 1.0)
        hip_roll = np.arccos(hip_roll_cos) * np.sign(hip_to_ank_roll_vec[1])

        leg_joint_pos = np.array(
            [
                hip_pitch,
                hip_roll,
                0.0,
                -knee,
                hip_roll,
                -ank_pitch,
                -hip_pitch,
                -hip_roll,
                0.0,
                knee,
                hip_roll,
                ank_pitch,
            ],
            dtype=np.float32,
        )

        return leg_joint_pos

    def get_qpos_ref(self, state_ref: ArrayType, path_frame=True) -> ArrayType:
        """Compute the reference joint positions (qpos) for the robot based on a given state reference.

        This function updates the default joint positions with the motor, joint, and passive joint positions
        from the state reference. It also adjusts the torso orientation and position based on the waist joint
        positions and the selected foot position.

        Args:
            state_ref (ArrayType): The reference state array containing motor, joint, and passive joint positions.
            path_frame (bool, optional): If True, the torso orientation is set relative to the path frame.
                If False, it is set relative to the global frame. Defaults to True.

        Returns:
            ArrayType: The updated joint positions (qpos) for the robot.
        """
        qpos = self.default_qpos.copy()

        motor_pos_ref = state_ref[13 : 13 + self.robot.nu]
        joint_pos_ref = state_ref[13 + self.robot.nu : 13 + 2 * self.robot.nu]
        qpos = inplace_update(qpos, 7 + self.mj_motor_indices, motor_pos_ref)
        qpos = inplace_update(qpos, 7 + self.mj_joint_indices, joint_pos_ref)

        passive_pos_ref = (
            state_ref[13 + self.robot.nu + self.passive_joint_indices]
            * self.passive_joint_signs
        )
        qpos = inplace_update(qpos, 7 + self.mj_passive_indices, passive_pos_ref)

        waist_joint_pos = state_ref[13 + self.robot.nu + self.waist_joint_indices]
        waist_euler = np.array([waist_joint_pos[0], 0.0, waist_joint_pos[1]])
        torso_quat = quat_inv(euler2quat(waist_euler))
        if path_frame:
            qpos = inplace_update(qpos, slice(3, 7), torso_quat)  # ignore the path quat
        else:
            qpos = inplace_update(
                qpos, slice(3, 7), quat_mult(state_ref[3:7], torso_quat)
            )

        data = self.forward(qpos)

        left_foot_pos = data.site_xpos[self.left_foot_site_id]
        right_foot_pos = data.site_xpos[self.right_foot_site_id]

        # Select the foot with the smaller z-coordinate
        foot_delta = np.where(
            left_foot_pos[2] < right_foot_pos[2],
            self.left_foot_center - left_foot_pos,
            self.right_foot_center - right_foot_pos,
        )
        # Compute the torso position adjustment
        torso_pos = self.torso_pos_init + foot_delta

        if path_frame:
            qpos = inplace_update(qpos, slice(0, 3), torso_pos)
        else:
            qpos = inplace_update(qpos, slice(0, 3), torso_pos + state_ref[:3])

        return qpos
