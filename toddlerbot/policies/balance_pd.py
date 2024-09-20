from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import interpolate_action

default_pose = np.array(
    [
        -0.6028545,
        -0.90198064,
        0.01840782,
        1.2379225,
        0.52615595,
        0.4985056,
        -1.1320779,
        0.5031457,
        -0.9372623,
        -0.248505,
        1.2179809,
        -0.35434943,
        -0.6473398,
        -1.1581556,
    ]
)


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos)

        if fixed_command is None:
            self.fixed_command = np.array([0.0], dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        # Indices for the pitch joints
        self.ctrl_x_indices = [
            robot.joint_ordering.index("left_hip_pitch"),
            robot.joint_ordering.index("left_knee_pitch"),
            robot.joint_ordering.index("left_ank_pitch"),
            robot.joint_ordering.index("right_hip_pitch"),
            robot.joint_ordering.index("right_knee_pitch"),
            robot.joint_ordering.index("right_ank_pitch"),
        ]
        self.ctrl_y_indices = [
            robot.joint_ordering.index("left_hip_roll"),
            robot.joint_ordering.index("left_ank_roll"),
            robot.joint_ordering.index("right_hip_roll"),
            robot.joint_ordering.index("right_ank_roll"),
        ]

        teleop_default_motor_pos = self.default_motor_pos.copy()
        arm_motor_slice = slice(
            robot.motor_ordering.index("left_sho_pitch"),
            robot.motor_ordering.index("right_wrist_roll") + 1,
        )
        teleop_default_motor_pos[arm_motor_slice] = default_pose

        self.motion_ref = BalanceReference(robot)

        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.com_pos_init: npt.NDArray[np.float32] | None = None

        self.joint_indices = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        if "fixed" not in self.name:
            # Disregard the free joint
            self.joint_indices -= 1

        self.q_start_idx = 0 if "fixed" in self.name else 7

        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            teleop_default_motor_pos,
            self.prep_duration,
            end_time=5.0,
        )

        # PD controller parameters
        self.kp = 2000.0
        self.kd = 0.0

        self.step_curr = 0
        self.previous_error = np.zeros(2, dtype=np.float32)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        command = self.fixed_command

        time_curr = self.step_curr * self.control_dt
        state_ref = self.motion_ref.get_state_ref(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            time_curr,
            command,
        )

        motor_angles = dict(zip(self.robot.motor_ordering, obs.motor_pos))
        for name in motor_angles:
            self.data.joint(name).qpos = motor_angles[name]

        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

        mujoco.mj_forward(self.model, self.data)
        com_pos = np.asarray(self.data.body(0).subtree_com, dtype=np.float32)
        com_jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, com_jacp, 0)

        if self.com_pos_init is None:
            self.com_pos_init = np.asarray(
                self.data.body(0).subtree_com, dtype=np.float32
            )

        error = com_pos[:2] - self.com_pos_init[:2]
        error_derivative = (error - self.previous_error) / self.control_dt
        self.previous_error = error

        ctrl = self.kp * error + self.kd * error_derivative

        # Update joint positions based on the PD controller command
        joint_pos = self.default_joint_pos.copy()
        # Define joint mappings for ctrl[0] and ctrl[1]

        # Update joint positions for ctrl[0]
        joint_pos[self.ctrl_x_indices] -= (
            ctrl[0]
            * com_jacp[0, self.q_start_idx + self.joint_indices[self.ctrl_x_indices]]
        )

        # Update joint positions for ctrl[1]
        joint_pos[self.ctrl_y_indices] -= (
            ctrl[1]
            * com_jacp[1, self.q_start_idx + self.joint_indices[self.ctrl_y_indices]]
        )

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        motor_target = np.asarray(
            self.motion_ref.override_motor_target(motor_target, state_ref)
        )
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1

        return motor_target
