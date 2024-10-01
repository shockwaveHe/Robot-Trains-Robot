import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import interpolate_action


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

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
        self.pitch_joint_signs = np.array([1, 1, 1, -1, -1, 1], dtype=np.float32)

        self.neck_yaw_idx = robot.joint_ordering.index("neck_yaw_driven")
        self.neck_pitch_idx = robot.joint_ordering.index("neck_pitch_driven")

        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.com_pos_init = np.asarray(self.data.body(0).subtree_com, dtype=np.float32)

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
        self.com_kp = np.array([2000, 2000], dtype=np.float32)
        self.com_kd = np.array([0, 0], dtype=np.float32)

        self.com_pos_error_prev = np.zeros(2, dtype=np.float32)
        self.step_curr = 0
        self.last_motor_target = self.default_motor_pos.copy()

    def reset(self):
        self.com_pos_error_prev = np.zeros(2, dtype=np.float32)
        self.step_curr = 0
        self.last_motor_target = self.default_motor_pos.copy()

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        # Get the motor angles from the observation
        # self.data.joint(0).qpos[3:7] = euler2quat(obs.euler)
        motor_angles = dict(zip(self.robot.motor_ordering, obs.motor_pos))
        for name in motor_angles:
            self.data.joint(name).qpos = motor_angles[name]

        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

        mujoco.mj_forward(self.model, self.data)

        # Get the center of mass position
        com_pos = np.asarray(self.data.body(0).subtree_com, dtype=np.float32)

        # PD controller on CoM position
        com_pos_error = com_pos[:2] - self.com_pos_init[:2]
        com_pos_error_d = (com_pos_error - self.com_pos_error_prev) / self.control_dt
        self.com_pos_error_prev = com_pos_error

        # print(f"com_pos: {com_pos}")
        # print(f"com_pos_init: {self.com_pos_init}")
        # print(f"com_pos_error: {com_pos_error}")

        ctrl_com = self.com_kp * com_pos_error + self.com_kd * com_pos_error_d

        joint_target = self.plan()

        # print(f"hip_pitch_target: {joint_target[6]:.4f} {joint_target[12]:.4f}")

        com_jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, com_jacp, 0)

        # Update joint positions based on the PD controller command
        joint_target[self.pitch_joint_indicies] -= (
            ctrl_com[0]
            * com_jacp[
                0, self.q_start_idx + self.joint_indices[self.pitch_joint_indicies]
            ]
        )
        joint_target[self.roll_joint_indicies] -= (
            ctrl_com[1]
            * com_jacp[
                1, self.q_start_idx + self.joint_indices[self.roll_joint_indicies]
            ]
        )

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_target))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)
        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        # print(f"hip_pitch: {motor_target[6]:.4f} {motor_target[12]:.4f}")

        self.step_curr += 1
        self.last_motor_target = motor_target

        return motor_target

    def plan(self) -> npt.NDArray[np.float32]:
        joint_target = self.default_joint_pos.copy()
        joint_angles = self.robot.motor_to_joint_angles(
            dict(zip(self.robot.motor_ordering, self.last_motor_target))
        )
        joint_target[self.neck_yaw_idx] = joint_angles["neck_yaw_driven"]
        joint_target[self.neck_pitch_idx] = joint_angles["neck_pitch_driven"]

        return joint_target
