from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.motion_reference.zmp_planner import ZMPPlanner
from toddlerbot.sim.robot import Robot


class WalkZMPReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        use_jax: bool = False,
        default_joint_pos: Optional[jax.Array] = None,
        default_joint_vel: Optional[jax.Array] = None,
        max_knee_pitch: float = jnp.pi / 3,
        double_support_phase: float = 0.5,
        single_support_phase: float = 1.0,
        stride_size: float = 0.5,
        zmp_y_offset: float = 0.05,
        num_steps: int = 100,
    ):
        super().__init__("walk", "periodic", robot, use_jax)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.knee_pitch_default = 0.0
        else:
            self.knee_pitch_default = self.default_joint_pos[
                self.get_joint_idx("left_knee_pitch")
            ]

        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )
        self.com_z = self.robot.config["general"]["offsets"]["torso_z"]
        self.max_knee_pitch = max_knee_pitch
        self.double_support_phase = double_support_phase
        self.single_support_phase = single_support_phase
        self.stride_size = stride_size
        self.zmp_y_offset = zmp_y_offset
        self.num_steps = num_steps

        self.zmp_planner = ZMPPlanner()

    def reset(self):
        pass

    def get_zmp_ref(
        self,
        path_pos: jax.Array,
        path_quat: jax.Array,
        phase: float,
        command: jax.Array,
        dt: float = 0.01,
    ):
        desired_zmps: List[jax.Array] = []
        for i in range(self.num_steps):
            footstep = jnp.array(  # type: ignore
                [i * self.stride_size, (-1) ** (i + 1) * self.zmp_y_offset]
            )
            if i == 0 or i == self.num_steps - 1:
                footstep = footstep.at[1].set(0)  # type: ignore

            desired_zmps.append(footstep)
            desired_zmps.append(footstep)

        time_list = jnp.array(  # type: ignore
            [0, self.single_support_phase]
            + [
                self.double_support_phase,
                self.single_support_phase,
            ]
            * (self.num_steps - 1),
            dtype=jnp.float32,
        )
        time_steps = jnp.cumsum(time_list)  # type: ignore

        self.zmp_planner.plan(
            time_steps,
            desired_zmps,
            jnp.array([0, 0, 0, 0], dtype=jnp.float32),  # type: ignore
            self.com_z,
            Qy=jnp.eye(2, dtype=jnp.float32),  # type: ignore
            R=jnp.eye(2, dtype=jnp.float32) * 0.1,  # type: ignore
        )

        x0 = jnp.array([path_pos[0], path_pos[1], command[0], command[1]])  # type: ignore

        N = int((time_steps[-1] - time_steps[0]) / dt)

        time = time_steps[0] + jnp.arange(N) * dt
        com_pos = jax.vmap(self.zmp_planner.get_nominal_com)(time)

        traj = {
            "time": jnp.zeros(N, dtype=jnp.float32),  # type: ignore
            "x": jnp.zeros((N, 4), dtype=jnp.float32),  # type: ignore
            "u": jnp.zeros((N, 2), dtype=jnp.float32),  # type: ignore
            "cop": jnp.zeros((N, 2), dtype=jnp.float32),  # type: ignore
            "desired_zmp": jnp.zeros((N, 2), dtype=jnp.float32),  # type: ignore
            "nominal_com": jnp.zeros((N, 6), dtype=jnp.float32),  # type: ignore
        }

        for i in range(3):
            traj["time"] = traj["time"].at[i].set(time_steps[0] + i * dt)
            if i == 0:
                traj["x"] = traj["x"].at[i, :].set(x0)
            else:
                xd = jnp.hstack((traj["x"][i - 1, 2:], traj["u"][i - 1, :]))  # type: ignore
                traj["x"] = traj["x"].at[i, :].set(traj["x"][i - 1, :] + xd * dt)

            traj["u"] = (
                traj["u"]
                .at[i, :]
                .set(
                    self.zmp_planner.get_optim_com_acc(traj["time"][i], traj["x"][i, :])
                )
            )

            traj["cop"] = (
                traj["cop"]
                .at[i, :]
                .set(self.zmp_planner.com_acc_to_cop(traj["x"][i, :], traj["u"][i, :]))
            )

            traj["desired_zmp"] = (
                traj["desired_zmp"]
                .at[i, :]
                .set(self.zmp_planner.get_desired_zmp(traj["time"][i]))
            )

            traj["nominal_com"] = (
                traj["nominal_com"]
                .at[i, :2]
                .set(self.zmp_planner.get_nominal_com(traj["time"][i]))
            )
            traj["nominal_com"] = (
                traj["nominal_com"]
                .at[i, 2:4]
                .set(self.zmp_planner.get_nominal_com_vel(traj["time"][i]))
            )
            traj["nominal_com"] = (
                traj["nominal_com"]
                .at[i, 4:]
                .set(self.zmp_planner.get_nominal_com_acc(traj["time"][i]))
            )

        with open("jax_results.txt", "w") as file:
            file.write("Time:\n" + str(traj["time"]) + "\n\n")
            file.write("Desired ZMP:\n" + str(traj["desired_zmp"]) + "\n\n")
            file.write("Nominal COM:\n" + str(traj["nominal_com"]) + "\n\n")
            file.write("Center of Pressure (COP):\n" + str(traj["cop"]) + "\n\n")
            file.write("State Vector (x):\n" + str(traj["x"]) + "\n\n")
            file.write("Control Inputs (u):\n" + str(traj["u"]) + "\n\n")

        return traj

    def get_state_ref(
        self,
        path_pos: jax.Array,
        path_quat: jax.Array,
        phase: Optional[float | jax.Array] = None,
        command: Optional[jax.Array] = None,
    ) -> jax.Array:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        linear_vel = jnp.array([command[0], command[1], 0.0], dtype=jnp.float32)
        angular_vel = jnp.array([0.0, 0.0, command[2]], dtype=jnp.float32)

        sin_phase_signal = jnp.sin(2 * jnp.pi * phase)
        signal_left = jnp.clip(sin_phase_signal, 0, None)
        signal_right = jnp.clip(sin_phase_signal, None, 0)

        if self.default_joint_pos is None:
            joint_pos = jnp.zeros(self.num_joints, dtype=jnp.float32)
        else:
            joint_pos = self.default_joint_pos.copy()

        if self.default_joint_vel is None:
            joint_vel = jnp.zeros(self.num_joints, dtype=jnp.float32)
        else:
            joint_vel = self.default_joint_vel.copy()

        left_leg_angles = self.calculate_leg_angles(signal_left, True)
        right_leg_angles = self.calculate_leg_angles(signal_right, False)

        leg_angles = {**left_leg_angles, **right_leg_angles}

        if self.use_jax:
            indices = jnp.array([self.get_joint_idx(name) for name in leg_angles])
            angles = jnp.array(list(leg_angles.values()))
            joint_pos = joint_pos.at[indices].set(angles)
        else:
            for name, angle in leg_angles.items():
                joint_pos[self.get_joint_idx(name)] = angle

        double_support_mask = jnp.abs(sin_phase_signal) < self.double_support_phase
        joint_pos = jnp.where(double_support_mask, self.default_joint_pos, joint_pos)

        stance_mask = jnp.zeros(2, dtype=jnp.float32)
        if self.use_jax:
            stance_mask = stance_mask.at[0].set(jnp.any(sin_phase_signal >= 0))
            stance_mask = stance_mask.at[1].set(jnp.any(sin_phase_signal < 0))
            stance_mask = jnp.where(double_support_mask, 1, stance_mask)
        else:
            stance_mask[0] = jnp.any(sin_phase_signal >= 0)
            stance_mask[1] = jnp.any(sin_phase_signal < 0)
            stance_mask[double_support_mask] = 1

        return jnp.concatenate(
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            )
        )

    def calculate_leg_angles(self, signal: jax.Array, is_left: bool):
        knee_angle = jnp.abs(
            signal * (self.max_knee_pitch - self.knee_pitch_default)
            + (2 * int(is_left) - 1) * self.knee_pitch_default
        )
        ank_pitch_angle = jnp.arctan2(
            jnp.sin(knee_angle), jnp.cos(knee_angle) + self.shin_thigh_ratio
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return {
                "left_hip_pitch": -hip_pitch_angle,
                "left_knee_pitch": knee_angle,
                "left_ank_pitch": -ank_pitch_angle,
            }
        else:
            return {
                "right_hip_pitch": hip_pitch_angle,
                "right_knee_pitch": -knee_angle,
                "right_ank_pitch": -ank_pitch_angle,
            }


if __name__ == "__main__":
    from toddlerbot.utils.math_utils import round_to_sig_digits

    robot = Robot("toddlerbot")
    walk_ref = WalkZMPReference(robot, max_knee_pitch=0.523599)
    left_leg_angles = walk_ref.calculate_leg_angles(
        jnp.ones(1, dtype=jnp.float32), True
    )
    left_ank_act = robot.ankle_ik([0.0, left_leg_angles["left_ank_pitch"].item()])
    print(left_leg_angles)
    print([round_to_sig_digits(x, 6) for x in left_ank_act])
