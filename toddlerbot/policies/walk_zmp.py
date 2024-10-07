from typing import List, Optional

import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.algorithms.zmp.zmp_walk import ZMPWalk
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import interpolate_action


class WalkZMPPolicy(BasePolicy, policy_name="walk_zmp"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        cycle_time: float = 0.72,
    ):
        super().__init__(name, robot, init_motor_pos)

        if fixed_command is None:
            self.fixed_command = np.array([0.1, 0.0, 0.0], dtype=np.float32)
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

        joint_groups = np.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_joint_indices = np.arange(robot.nu)[joint_groups == "leg"]

        # Indices for the pitch joints
        self.pitch_joint_indicies = [
            robot.joint_ordering.index("left_hip_pitch"),
            robot.joint_ordering.index("left_knee_pitch"),
            robot.joint_ordering.index("left_ank_pitch"),
            robot.joint_ordering.index("right_hip_pitch"),
            robot.joint_ordering.index("right_knee_pitch"),
            robot.joint_ordering.index("right_ank_pitch"),
        ]
        self.roll_joint_indicies = [
            robot.joint_ordering.index("left_hip_roll"),
            robot.joint_ordering.index("left_ank_roll"),
            robot.joint_ordering.index("right_hip_roll"),
            robot.joint_ordering.index("right_ank_roll"),
        ]

        self.zmp_walk = ZMPWalk(robot, cycle_time, control_dt=self.control_dt)
        self.com_ref, self.leg_joint_pos_ref, stance_mask_ref = self.zmp_walk.plan(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            self.fixed_command,
            total_time=60.0,
        )

        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

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

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        # PD controller parameters
        self.kp = np.array([200, 200], dtype=np.float32)
        self.kd = np.array([0, 0], dtype=np.float32)

        self.step_curr = 0
        self.previous_error = np.zeros(2, dtype=np.float32)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        motor_angles = dict(zip(self.robot.motor_ordering, obs.motor_pos))
        for name in motor_angles:
            self.data.joint(name).qpos = motor_angles[name]

        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

        mujoco.mj_forward(self.model, self.data)
        com_jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, com_jacp, 0)

        com_pos_ref = self.com_ref[self.step_curr]
        error = obs.torso_pos[:2] - com_pos_ref[:2]
        error_derivative = (error - self.previous_error) / self.control_dt
        self.previous_error = error

        ctrl = self.kp * error + self.kd * error_derivative

        # Update joint positions based on the PD controller command
        joint_pos = self.default_joint_pos.copy()
        joint_pos[self.leg_joint_indices] = self.leg_joint_pos_ref[self.step_curr]

        # Update joint positions for ctrl[0]
        joint_pos[self.pitch_joint_indicies] -= (
            ctrl[0]
            * com_jacp[
                0, self.q_start_idx + self.joint_indices[self.pitch_joint_indicies]
            ]
        )

        # Update joint positions for ctrl[1]
        joint_pos[self.roll_joint_indicies] -= (
            ctrl[1]
            * com_jacp[
                1, self.q_start_idx + self.joint_indices[self.roll_joint_indicies]
            ]
        )

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1

        return motor_target
