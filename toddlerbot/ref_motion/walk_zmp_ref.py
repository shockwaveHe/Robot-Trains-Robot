import os
import pickle
from typing import List, Optional, Tuple

import jax
from tqdm import tqdm

from toddlerbot.algorithms.zmp.footstep_planner import FootStepPlanner
from toddlerbot.algorithms.zmp.zmp_planner import ZMPPlanner
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update, loop_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat2euler


class WalkZMPReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        command_ranges: List[List[float]],
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        stride_max: List[float] = [0.12, 0.04, np.pi / 16],
        single_double_ratio: float = 2.0,
        foot_step_height: float = 0.03,
        control_dt: float = 0.012,
        control_cost_Q: float = 1.0,
        control_cost_R: float = 0.1,
    ):
        super().__init__("walk_zmp", "periodic", robot)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        self.stride_max = np.array(stride_max, dtype=np.float32)  # type: ignore
        self.single_double_ratio = single_double_ratio
        self.footstep_height = foot_step_height
        self.control_dt = control_dt
        self.control_cost_Q = control_cost_Q
        self.control_cost_R = control_cost_R

        # TODO: Read from config
        self.com_z = 0.336
        self.foot_to_com_x = float(robot.data_dict["offsets"]["foot_to_com_x"])
        self.foot_to_com_y = float(robot.data_dict["offsets"]["foot_to_com_y"])

        if self.default_joint_pos is None:
            self.default_target_z = 0.0
        else:
            self.default_target_z = (
                float(robot.config["general"]["offsets"]["torso_z"]) - self.com_z
            )

        self.leg_joint_slice = slice(
            self.get_joint_idx("left_hip_yaw_driven"),
            self.get_joint_idx("right_ank_pitch") + 1,
        )

        self.footstep_planner = FootStepPlanner(self.stride_max, self.foot_to_com_y)
        self.zmp_planner = ZMPPlanner()

        self.lookup_table_path = os.path.join(
            "toddlerbot", "ref_motion", "walk_zmp_lookup_table.pkl"
        )
        self.build_lookup_table(command_ranges)

    # @profile()
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
        duration: Optional[float] = None,
    ) -> ArrayType:
        if phase is None:
            raise ValueError(f"phase is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        if duration is None:
            raise ValueError(f"duration is required for {self.name}")

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)  # type: ignore

        if self.default_joint_vel is None:
            joint_vel = np.zeros(len(self.robot.joint_ordering), dtype=np.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        is_zero_commmand = np.linalg.norm(command) < 1e-6  # type: ignore
        nearest_command_idx = np.argmin(  # type: ignore
            np.linalg.norm(self.lookup_keys - command, axis=1)  # type: ignore
        )
        idx = (phase / self.control_dt).astype(int)  # type: ignore

        joint_pos = self.default_joint_pos.copy()  # type: ignore
        joint_pos = np.where(  # type: ignore
            is_zero_commmand,  # type: ignore
            joint_pos,
            inplace_update(
                joint_pos,
                self.leg_joint_slice,
                self.leg_joint_pos_lookup[nearest_command_idx][idx],
            ),
        )
        stance_mask = np.where(  # type: ignore
            is_zero_commmand,  # type: ignore
            np.ones(2, dtype=np.float32),  # type: ignore
            self.stance_mask_lookup[nearest_command_idx][idx],
        )

        return np.concatenate(  # type: ignore
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

    def build_lookup_table(
        self,
        command_ranges: List[List[float]],
        interval: float = 0.1,
        duration: float = 10.0,
    ):
        """
        Precompute and store the trajectories for a range of commands.
        """
        if os.path.exists(self.lookup_table_path):
            with open(self.lookup_table_path, "rb") as f:
                lookup_keys, stance_mask_ref_list, leg_joint_pos_ref_list = pickle.load(
                    f
                )
        else:
            lookup_keys: List[Tuple[float, ...]] = []
            stance_mask_ref_list: List[ArrayType] = []
            leg_joint_pos_ref_list: List[ArrayType] = []

            path_pos = np.zeros(3, dtype=np.float32)  # type: ignore
            path_quat = np.array([1, 0, 0, 0], dtype=np.float32)  # type: ignore

            # Create linspace arrays for each command range
            linspaces = [
                np.arange(start, stop + interval, interval)  # type: ignore
                for start, stop in command_ranges
            ]
            # Create a meshgrid
            meshgrid = np.meshgrid(*linspaces, indexing="ij")  # type: ignore
            command_spectrum = np.stack(meshgrid, axis=-1).reshape(-1, 3)  # type: ignore
            # command_spectrum = np.array([[0.0, 0.0, 0.2]])
            for command in tqdm(command_spectrum, desc="Building Lookup Table"):
                if np.linalg.norm(command) < 1e-6:  # type: ignore
                    continue

                leg_joint_pos_ref, stance_mask_ref = self.plan(
                    path_pos, path_quat, command, duration
                )
                lookup_keys.append(tuple(map(float, command)))
                stance_mask_ref_list.append(stance_mask_ref)
                leg_joint_pos_ref_list.append(leg_joint_pos_ref)

            with open(self.lookup_table_path, "wb") as f:
                pickle.dump(
                    (lookup_keys, stance_mask_ref_list, leg_joint_pos_ref_list), f
                )

        self.lookup_keys = np.array(lookup_keys, dtype=np.float32)  # type: ignore
        num_commands = len(stance_mask_ref_list)
        num_total_steps_max = max(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list]
        )
        self.stance_mask_lookup = np.zeros(  # type: ignore
            (num_commands, num_total_steps_max, 2), dtype=np.float32
        )
        self.leg_joint_pos_lookup = np.zeros(  # type: ignore
            (num_commands, num_total_steps_max, 12), dtype=np.float32
        )
        for i, (stance_mask_ref, leg_joint_pos_ref) in enumerate(
            zip(stance_mask_ref_list, leg_joint_pos_ref_list)
        ):
            self.stance_mask_lookup = inplace_update(  # type: ignore
                self.stance_mask_lookup,
                (i, slice(None, len(stance_mask_ref))),
                stance_mask_ref,
            )
            self.leg_joint_pos_lookup = inplace_update(  # type: ignore
                self.leg_joint_pos_lookup,
                (i, slice(None, len(leg_joint_pos_ref))),
                leg_joint_pos_ref,
            )

        if os.environ.get("USE_JAX", "false") == "true":
            self.lookup_keys = jax.device_put(self.lookup_keys)  # type: ignore
            self.stance_mask_lookup = jax.device_put(self.stance_mask_lookup)  # type: ignore
            self.leg_joint_pos_lookup = jax.device_put(self.leg_joint_pos_lookup)  # type: ignore

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        command: ArrayType,
        duration: float,
    ):
        path_euler = quat2euler(path_quat)
        pose_curr = np.array(  # type: ignore
            [path_pos[0], path_pos[1], path_euler[2]], dtype=np.float32
        )

        spline_x, spline_y, spline_theta = self.sample_spline(
            pose_curr, command, duration
        )
        _, footsteps = self.footstep_planner.compute_steps(
            pose_curr,
            np.array([spline_x[-1], spline_y[-1], spline_theta[-1]], dtype=np.float32),  # type: ignore
            has_start=False,
            has_stop=False,
        )

        # import numpy

        # from toddlerbot.visualization.vis_plot import plot_footsteps

        # plot_footsteps(
        #     numpy.asarray(path, dtype=numpy.float32),
        #     numpy.array(
        #         [numpy.asarray(fs[:3]) for fs in footsteps], dtype=numpy.float32
        #     ),
        #     [int(fs[-1]) for fs in footsteps],
        #     (0.1, 0.05),
        #     self.foot_to_com_y,
        #     fig_size=(8, 8),
        #     title="Footsteps Planning",
        #     x_label="Position X",
        #     y_label="Position Y",
        #     save_config=False,
        #     save_path=".",
        #     file_name="footsteps.png",
        # )()

        num_footsteps = len(footsteps)
        double_support_phase = duration / (
            (num_footsteps - 1) * (1 + self.single_double_ratio) + 1
        )
        double_support_phase = (
            np.ceil(double_support_phase / self.control_dt) * self.control_dt  # type: ignore
        )
        single_support_phase = double_support_phase * self.single_double_ratio
        time_list = np.array(  # type: ignore
            [0, double_support_phase]
            + [single_support_phase, double_support_phase] * (num_footsteps - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)  # type: ignore
        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]

        x0 = np.array(  # type: ignore
            [path_pos[0], path_pos[1], 0.0, 0.0], dtype=np.float32
        )

        self.zmp_planner.plan(
            time_steps,
            desired_zmps,
            x0,
            self.com_z,
            Qy=np.eye(2, dtype=np.float32) * self.control_cost_Q,  # type: ignore
            R=np.eye(2, dtype=np.float32) * self.control_cost_R,  # type: ignore
        )

        def update_step(
            carry: Tuple[ArrayType, ArrayType], idx: int
        ) -> Tuple[Tuple[ArrayType, ArrayType], ArrayType]:
            x_traj, u_traj = carry
            t = time_steps[0] + idx * self.control_dt
            xd = np.hstack((x_traj[idx - 1, 2:], u_traj[idx - 1, :]))  # type: ignore
            x_traj = inplace_update(
                x_traj, idx, x_traj[idx - 1, :] + xd * self.control_dt
            )
            u = self.zmp_planner.get_optim_com_acc(t, x_traj[idx, :])
            u_traj = inplace_update(u_traj, idx, u)

            return (x_traj, u_traj), x_traj[idx]

        # Initialize the arrays
        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)  # type: ignore
        )
        x_traj = np.zeros((num_total_steps, 4), dtype=np.float32)  # type: ignore
        u_traj = np.zeros((num_total_steps, 2), dtype=np.float32)  # type: ignore
        # Set the initial conditions
        x_traj = inplace_update(x_traj, 0, x0)
        u_traj = inplace_update(
            u_traj,
            0,
            self.zmp_planner.get_optim_com_acc(time_steps[0], x0),  # type: ignore
        )
        x_traj = loop_update(update_step, x_traj, u_traj, (1, num_total_steps))

        (
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            stance_mask_ref,
        ) = self.compute_foot_trajectories(
            time_steps,
            np.repeat(np.stack(footsteps), 2, axis=0),  # type: ignore
        )

        leg_joint_pos_ref = self.solve_ik(
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            x_traj[:, :2],
        )

        return leg_joint_pos_ref, stance_mask_ref

    def sample_spline(
        self,
        pose_curr: ArrayType,
        command: ArrayType,
        duration: float,
    ) -> Tuple[ArrayType, ...]:
        # Linear velocities in local frame
        v_x, v_y, v_yaw = command  # type: ignore = command

        timesteps = np.linspace(  # type: ignore
            0,
            duration,
            int(np.ceil(duration / self.control_dt)),  # type: ignore
            dtype=np.float32,
        )
        yaw_traj = pose_curr[2] + v_yaw * timesteps

        # Calculate the differences between consecutive timesteps
        dt = np.diff(np.concatenate((np.zeros(1, dtype=np.float32), timesteps)))  # type: ignore

        # Calculate x and y increments
        delta_x = (v_x * np.cos(yaw_traj) - v_y * np.sin(yaw_traj)) * dt  # type: ignore
        delta_y = (v_x * np.sin(yaw_traj) + v_y * np.cos(yaw_traj)) * dt  # type: ignore

        # Compute the full x and y trajectories by cumulative summing the increments
        x_traj = np.cumsum(delta_x)  # type: ignore
        y_traj = np.cumsum(delta_y)  # type: ignore

        last_x, last_y, last_theta = pose_curr
        sampled_x = [last_x]
        sampled_y = [last_y]
        sampled_theta = [last_theta]

        for i in range(1, len(timesteps)):
            x_dist = (x_traj[i] - last_x) * np.cos(last_theta) - (  # type: ignore
                y_traj[i] - last_y
            ) * np.sin(last_theta)  # type: ignore
            y_dist = (x_traj[i] - last_x) * np.sin(last_theta) + (  # type: ignore
                y_traj[i] - last_y
            ) * np.cos(last_theta)  # type: ignore
            l1_distance = np.abs(  # type: ignore
                np.array(  # type: ignore
                    [x_dist, y_dist, yaw_traj[i] - last_theta]
                )
            )

            if (
                np.any(l1_distance >= self.stride_max)  # type: ignore
                or i == len(timesteps) - 1
            ):
                sampled_x.append(x_traj[i])
                sampled_y.append(y_traj[i])
                sampled_theta.append(yaw_traj[i])
                last_x, last_y, last_theta = x_traj[i], y_traj[i], yaw_traj[i]

        return sampled_x, sampled_y, sampled_theta  # type: ignore

    def compute_foot_trajectories(
        self, time_steps: ArrayType, footsteps: List[ArrayType]
    ) -> Tuple[ArrayType, ...]:
        offset = np.array(  # type: ignore
            [
                -np.sin(footsteps[0][2]) * self.foot_to_com_y,  # type: ignore
                np.cos(footsteps[0][2]) * self.foot_to_com_y,  # type: ignore
            ]
        )
        last_pos = np.concatenate(  # type: ignore
            [
                footsteps[0][:2] + offset,
                np.zeros(1, dtype=np.float32),  # type: ignore
                footsteps[0][:2] - offset,
                np.zeros(1, dtype=np.float32),  # type: ignore
            ]
        )
        last_ori = np.array(  # type: ignore
            [0.0, 0.0, footsteps[0][2], 0.0, 0.0, footsteps[0][2]], dtype=np.float32
        )

        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)  # type: ignore
        )
        left_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)  # type: ignore
        left_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)  # type: ignore
        right_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)  # type: ignore
        right_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)  # type: ignore
        stance_mask_traj = np.zeros((num_total_steps, 2), dtype=np.float32)  # type: ignore
        step_curr = 0
        for i in range(len(time_steps) - 1):
            num_steps = round((time_steps[i + 1] - time_steps[i]) / self.control_dt)
            if num_steps + step_curr > num_total_steps:
                num_steps = num_total_steps - step_curr

            stance_mask = np.tile(np.ones(2, dtype=np.float32), (num_steps, 1))  # type: ignore
            if i % 2 == 0:  # Double support
                foot_pos_traj = np.tile(last_pos, (num_steps, 1))  # type: ignore
                foot_ori_traj = np.tile(last_ori, (num_steps, 1))  # type: ignore
            else:
                support_leg_curr = int(footsteps[i][-1])
                support_leg_next = int(footsteps[i + 1][-1])
                if support_leg_curr == 2:
                    current_pos = last_pos.copy()
                    current_ori = last_ori.copy()
                    swing_leg = support_leg_next
                else:
                    current_pos = inplace_update(
                        last_pos,
                        slice(support_leg_curr * 3, support_leg_curr * 3 + 2),
                        footsteps[i][:2],
                    )
                    current_ori = inplace_update(
                        last_ori,
                        support_leg_curr * 3 + 2,
                        footsteps[i][2],
                    )
                    swing_leg = 1 - support_leg_curr

                if support_leg_next == 2:
                    offset = np.array(  # type: ignore
                        [
                            -np.sin(footsteps[i][2]) * self.foot_to_com_y,  # type: ignore
                            np.cos(footsteps[i][2]) * self.foot_to_com_y,  # type: ignore
                        ]
                    ) * (-1 if support_leg_curr == 1 else 1)
                else:
                    offset = np.zeros(2, dtype=np.float32)  # type: ignore

                target_pos = inplace_update(
                    current_pos,
                    slice(swing_leg * 3, swing_leg * 3 + 2),
                    footsteps[i + 1][:2] + offset,
                )
                target_ori = inplace_update(
                    current_ori, swing_leg * 3 + 2, footsteps[i + 1][2]
                )
                last_pos = target_pos.copy()
                last_ori = target_ori.copy()

                up_delta = self.footstep_height / (num_steps // 2 - 1)
                up_traj = up_delta * np.concatenate(  # type: ignore
                    (
                        np.arange(num_steps // 2, dtype=np.float32),  # type: ignore
                        np.arange(  # type: ignore
                            num_steps - num_steps // 2 - 1, -1, -1, dtype=np.float32
                        ),
                    )
                )
                pos_delta = (target_pos - current_pos) / num_steps
                foot_pos_traj = current_pos + pos_delta * np.arange(num_steps)[:, None]  # type: ignore
                foot_pos_traj = inplace_update(
                    foot_pos_traj,  # type: ignore
                    (slice(None), swing_leg * 3 + 2),
                    up_traj,
                )

                # TODO: check this
                ori_delta = (target_ori - current_ori) / num_steps
                foot_ori_traj = current_ori + ori_delta * np.arange(num_steps)[:, None]  # type: ignore
                foot_ori_traj = inplace_update(
                    foot_ori_traj,  # type: ignore
                    (slice(None), swing_leg * 3 + 2),
                    np.zeros(num_steps, dtype=np.float32),  # type: ignore
                )
                stance_mask = inplace_update(stance_mask, (slice(None), swing_leg), 0)  # type: ignore

            slice_curr = slice(step_curr, step_curr + num_steps)
            left_foot_pos_traj = inplace_update(
                left_foot_pos_traj, slice_curr, foot_pos_traj[:, :3]
            )
            left_foot_ori_traj = inplace_update(
                left_foot_ori_traj, slice_curr, foot_ori_traj[:, :3]
            )
            right_foot_pos_traj = inplace_update(
                right_foot_pos_traj, slice_curr, foot_pos_traj[:, 3:]
            )
            right_foot_ori_traj = inplace_update(
                right_foot_ori_traj, slice_curr, foot_ori_traj[:, 3:]
            )
            stance_mask_traj = inplace_update(stance_mask_traj, slice_curr, stance_mask)

            step_curr += num_steps

        return (
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            stance_mask_traj,
        )

    def solve_ik(
        self,
        left_foot_pos_traj: ArrayType,
        left_foot_ori_traj: ArrayType,
        right_foot_pos_traj: ArrayType,
        right_foot_ori_traj: ArrayType,
        com_pos_traj: ArrayType,
    ):
        com_pos_traj_padded = np.hstack(  # type: ignore
            [com_pos_traj, np.zeros((com_pos_traj.shape[0], 1))]  # type: ignore
        )
        left_foot_adjusted_pos = (
            left_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, self.foot_to_com_y, 0], dtype=np.float32)  # type: ignore
        )
        right_foot_adjusted_pos = (
            right_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, -self.foot_to_com_y, 0], dtype=np.float32)  # type: ignore
        )

        left_leg_joint_pos_traj = self.foot_ik(
            left_foot_adjusted_pos,  # type: ignore
            left_foot_ori_traj,
            side="left",
        )
        right_leg_joint_pos_traj = self.foot_ik(
            right_foot_adjusted_pos,  # type: ignore
            right_foot_ori_traj,
            side="right",
        )

        # Combine the results for left and right legs
        leg_joint_pos_traj = np.hstack(  # type: ignore
            [left_leg_joint_pos_traj, right_leg_joint_pos_traj]
        )

        return leg_joint_pos_traj

    def foot_ik(
        self,
        target_foot_pos: ArrayType,
        target_foot_ori: ArrayType = np.zeros(3, dtype=np.float32),  # type: ignore
        side: str = "left",
    ) -> ArrayType:
        target_x = target_foot_pos[:, 0]
        target_y = target_foot_pos[:, 1]
        target_z = target_foot_pos[:, 2]
        ank_roll = target_foot_ori[:, 0]
        ank_pitch = target_foot_ori[:, 1]
        hip_yaw = target_foot_ori[:, 2]

        offsets = self.robot.data_dict["offsets"]

        transformed_x = target_x * np.cos(hip_yaw) + target_y * np.sin(hip_yaw)  # type: ignore
        transformed_y = -target_x * np.sin(hip_yaw) + target_y * np.cos(hip_yaw)  # type: ignore
        transformed_z = (
            offsets["hip_pitch_to_knee_z"]
            + offsets["knee_to_ank_pitch_z"]
            - target_z
            - self.default_target_z
        )

        hip_roll = np.arctan2(  # type: ignore
            transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"]
        )

        leg_projected_yz_length = np.sqrt(transformed_y**2 + transformed_z**2)  # type: ignore
        leg_length = np.sqrt(transformed_x**2 + leg_projected_yz_length**2)  # type: ignore
        leg_pitch = np.arctan2(transformed_x, leg_projected_yz_length)  # type: ignore
        hip_disp_cos = (
            leg_length**2
            + offsets["hip_pitch_to_knee_z"] ** 2
            - offsets["knee_to_ank_pitch_z"] ** 2
        ) / (2 * leg_length * offsets["hip_pitch_to_knee_z"])
        hip_disp = np.arccos(np.clip(hip_disp_cos, -1.0, 1.0))  # type: ignore
        ank_disp = np.arcsin(  # type: ignore
            offsets["hip_pitch_to_knee_z"]
            / offsets["knee_to_ank_pitch_z"]
            * np.sin(hip_disp)  # type: ignore
        )
        hip_pitch = -leg_pitch - hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch += knee_pitch + hip_pitch

        if side == "left":
            return np.vstack(  # type: ignore
                [
                    -hip_yaw,
                    -hip_roll,
                    hip_pitch,
                    knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]
            ).T
        else:
            return np.vstack(  # type: ignore
                [
                    -hip_yaw,
                    hip_roll,
                    -hip_pitch,
                    -knee_pitch,  # type: ignore
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]  # type: ignore
            ).T
