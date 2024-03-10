import argparse
import copy
import random
from typing import List, Optional, Tuple

import numpy as np

from toddlerbot.control.zmp_preview_control import *
from toddlerbot.planning.foot_step_planner import *
from toddlerbot.sim.mujoco_sim import MujoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.tasks.walking_configs import *
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.vis_planning import *
from toddlerbot.utils.vis_plot import *

random.seed(0)


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(
        self,
        robot: HumanoidRobot,
        config: WalkingConfig,
        com_pos_init: np.ndarray,
        joint_angles: List[float],
    ):
        self.robot = robot
        self.config = config
        self.fs_steps = round(config.plan_t_step / config.control_dt)
        self.x_offset_com_to_foot = com_pos_init[0]
        self.y_offset_com_to_foot = robot.config.offsets["y_offset_com_to_foot"]

        plan_params = FootStepPlanParameters(
            max_stride=config.plan_max_stride,
            t_step=config.plan_t_step,
            y_offset_com_to_foot=self.y_offset_com_to_foot,
        )
        self.fsp = FootStepPlanner(plan_params)

        control_params = ZMPPreviewControlParameters(
            com_z=robot.config.com_z - config.squat_height,
            dt=config.control_dt,
            t_preview=config.control_t_preview,
            Q_val=config.control_cost_Q_val,
            R_val=config.control_cost_R_val,
            x_offset_com_to_foot=self.x_offset_com_to_foot,
            y_disp_zmp=config.y_offset_zmp - self.y_offset_com_to_foot,
        )
        self.pc = ZMPPreviewController(control_params)

        self.curr_pose = None
        self.com_curr = np.concatenate(
            [com_pos_init[None, :2], np.zeros((2, 2))], axis=0
        )
        self.joint_angles = joint_angles

        self.idx = 0

        self.zmp_ref_record = []
        self.zmp_traj_record = []
        self.com_traj_record = []

        self.foot_steps = []
        self.com_traj = []

        self.left_up = self.right_up = 0.0
        # Assume the initial state is the canonical pose
        self.theta_curr = 0.0
        self.left_pos = np.array([0.0, self.y_offset_com_to_foot, 0.0])
        self.right_pos = np.array([0.0, -self.y_offset_com_to_foot, 0.0])

    def plan_and_control(
        self,
        curr_pose: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        com_pos_curr: Optional[np.ndarray] = None,
        use_feedback: bool = False,
    ) -> List[FootStep]:
        if self.curr_pose is not None and not use_feedback:
            curr_pose = self.curr_pose

        path, self.foot_steps = self.fsp.compute_steps(curr_pose, target_pose)

        if use_feedback:
            com_curr = np.concatenate(
                [com_pos_curr[None, :2], np.zeros((2, 2))], axis=0
            )
        else:
            com_curr = self.com_curr

        # Update the com trajectory based on the foot steps.
        zmp_ref, zmp_traj, com_traj, self.com_curr = self.pc.compute_com_traj(
            com_curr, self.foot_steps
        )
        self.com_traj = com_traj

        self.zmp_ref_record.extend(zmp_ref)
        self.zmp_traj_record.extend(zmp_traj)
        self.com_traj_record.extend(com_traj)

        return path, self.foot_steps, self.com_traj

    def solve_joint_angles(self) -> Tuple[List[float], List[float], int]:
        if len(self.com_traj) == 0:
            return "finished", self.joint_angles

        if abs(self.idx * self.config.control_dt - self.foot_steps[1].time) < 1e-6:
            self.foot_steps.pop(0)

            fs_curr = self.foot_steps[0]
            fs_next = self.foot_steps[1]
            self.theta_delta = (fs_next.position[2] - fs_curr.position[2]) / (
                self.fs_steps / 2
            )
            if fs_curr.support_leg == "left":
                if fs_next.support_leg == "right":
                    self.right_pos_target = fs_next.position
                else:
                    self.right_pos_target = np.array(
                        [
                            fs_next.position[0]
                            + np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            - np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[2],
                        ]
                    )

                self.right_pos_delta = (self.right_pos_target - self.right_pos) / (
                    self.fs_steps / 2
                )
            elif fs_curr.support_leg == "right":
                if fs_next.support_leg == "left":
                    self.left_pos_target = fs_next.position
                else:
                    self.left_pos_target = np.array(
                        [
                            fs_next.position[0]
                            - np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            + np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[2],
                        ]
                    )

                self.left_pos_delta = (self.left_pos_target - self.left_pos) / (
                    self.fs_steps / 2
                )

        com_pos = self.com_traj.pop(0)
        self._compute_foot_offset(com_pos)

        self.joint_angles = self.robot.solve_ik(
            *self._compute_foot_pos(com_pos),
            self.joint_angles,
        )
        self.idx += 1

        return "walking", self.joint_angles

    def _compute_foot_offset(self, com_pos):
        idx_curr = self.idx % self.fs_steps
        up_start_idx = round(self.fs_steps / 4)
        up_end_idx = round(self.fs_steps / 2)
        up_period = up_end_idx - up_start_idx

        up_delta = self.config.foot_step_height / up_period
        support_leg = self.foot_steps[0].support_leg

        # Up or down foot movement
        if up_start_idx < idx_curr <= up_end_idx:
            if support_leg == "right":
                self.left_up += up_delta
            elif support_leg == "left":
                self.right_up += up_delta
        else:
            if support_leg == "right":
                self.left_up = max(self.left_up - up_delta, 0.0)
            elif support_leg == "left":
                self.right_up = max(self.right_up - up_delta, 0.0)

        # Move foot in the axes of x, y, theta
        if idx_curr > up_start_idx + up_period * 2:
            if support_leg == "right":
                self.left_pos = self.left_pos_target.copy()
            elif support_leg == "left":
                self.right_pos = self.right_pos_target.copy()
        elif idx_curr > up_start_idx:
            if support_leg == "right":
                self.left_pos += self.left_pos_delta
                self.theta_curr += self.theta_delta
            elif support_leg == "left":
                self.right_pos += self.right_pos_delta
                self.theta_curr += self.theta_delta

    def _compute_foot_pos(self, com_pos):
        self.theta_curr = 0.0
        left_foot_theta = self.theta_curr - self.left_pos[2]
        right_foot_theta = self.theta_curr - self.right_pos[2]
        print(
            f"theta_curr: {self.theta_curr}, left_foot_theta: {left_foot_theta}, right_foot_theta: {right_foot_theta}"
        )
        left_foot_ori = [0, 0, left_foot_theta]
        right_foot_ori = [0, 0, right_foot_theta]

        left_offset = self.left_pos[:2] - com_pos
        right_offset = self.right_pos[:2] - com_pos

        left_foot_pos = [
            left_offset[0]
            + np.cos(self.theta_curr) * self.x_offset_com_to_foot
            + np.sin(self.theta_curr) * self.y_offset_com_to_foot,
            left_offset[1]
            + np.sin(self.theta_curr) * self.x_offset_com_to_foot
            - np.cos(self.theta_curr) * self.y_offset_com_to_foot,
            self.config.squat_height + self.left_up,
        ]

        right_foot_pos = [
            right_offset[0]
            + np.cos(self.theta_curr) * self.x_offset_com_to_foot
            - np.sin(self.theta_curr) * self.y_offset_com_to_foot,
            right_offset[1]
            + np.sin(self.theta_curr) * self.x_offset_com_to_foot
            + np.cos(self.theta_curr) * self.y_offset_com_to_foot,
            self.config.squat_height + self.right_up,
        ]

        # if self.idx == 0:
        #     print("stop")

        return left_foot_pos, left_foot_ori, right_foot_pos, right_foot_ori


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--use-feedback",
        action="store_true",
        default=False,
        help="Whether to use feedback control or not.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)
    if args.sim == "pybullet":
        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        sim = MujoCoSim(robot)
    else:
        raise ValueError("Unknown simulator")

    config = walking_configs[f"{args.robot_name}_{args.sim}"]

    joint_angles = sim.initialize_joint_angles(robot)
    if robot.name == "robotis_op3":
        joint_angles["l_sho_roll"] = np.pi / 2
        joint_angles["r_sho_roll"] = -np.pi / 2
    elif robot.name == "base":
        joint_angles["left_sho_roll"] = -np.pi / 2
        joint_angles["right_sho_roll"] = np.pi / 2

    com_pos_init = sim.get_com(robot)
    walking = Walking(robot, config, com_pos_init, joint_angles)

    torso_pos_init, torso_mat_init = sim.get_torso_pose(robot)
    curr_pose = np.concatenate(
        [torso_pos_init[:2], [np.arctan2(torso_mat_init[1, 0], torso_mat_init[0, 0])]]
    )
    target_pose = config.target_pose_init

    path, foot_steps, com_traj = walking.plan_and_control(
        curr_pose, target_pose, com_pos_init
    )
    foot_steps_vis = copy.deepcopy(foot_steps)

    # TODO: fix the small y_offset and let the body rotate with the hip
    # TODO: add the feedback control and the next plan
    # TODO: clean up the code

    joint_angle_errors = []
    com_traj_record = []

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx, path, foot_steps, com_traj, joint_angles):
        sim_step_idx += 1
        if args.sim == "mujoco":
            joint_angles_error = sim.get_joint_angles_error(robot, joint_angles)
            joint_angle_errors.append(np.linalg.norm(list(joint_angles_error.values())))

        if sim_step_idx >= config.actuator_steps:
            sim_step_idx = 0
            status, joint_angles = walking.solve_joint_angles()
            print(f"joint_angles: {round_floats(list(joint_angles.values()), 6)}")

            torso_pos, torso_mat = sim.get_torso_pose(robot)
            torso_mat_delta = torso_mat @ torso_mat_init.T
            torso_theta = np.arctan2(torso_mat_delta[1, 0], torso_mat_delta[0, 0])
            print(f"torso_pos: {torso_pos}, torso_theta: {torso_theta}")

            com_pos = [
                torso_pos[0] + np.cos(torso_theta) * walking.x_offset_com_to_foot,
                torso_pos[1] + np.sin(torso_theta) * walking.x_offset_com_to_foot,
            ]
            com_traj_record.append(com_pos)

            if status == "finished":
                tracking_error = np.array(target_pose) - np.array(
                    [*torso_pos[:2], torso_theta]
                )
                print(f"Tracking error: {round_floats(tracking_error, 6)}")

        sim.set_joint_angles(robot, joint_angles)

        return sim_step_idx, path, foot_steps, com_traj, joint_angles

    try:
        sim.simulate(
            step_func,
            (0, path, foot_steps, com_traj, joint_angles),
            args.sleep_time,
            vis_flags=["foot_steps", "com_traj", "path", "torso"],
        )
    finally:
        if args.sim == "mujoco":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_aspect("equal")

            draw_footsteps(
                path,
                foot_steps_vis,
                robot.config.foot_size[:2],
                robot.config.offsets["y_offset_com_to_foot"],
                title="Footsteps Planning",
                save_path="results/plots",
                time_suffix="",
                ax=ax,
            )()

            zmp_ref_record_x = [record[0] for record in walking.zmp_ref_record]
            zmp_ref_record_y = [record[1] for record in walking.zmp_ref_record]
            zmp_traj_record_x = [record[0] for record in walking.zmp_traj_record]
            zmp_traj_record_y = [record[1] for record in walking.zmp_traj_record]
            com_ref_record_x = [record[0] for record in walking.com_traj_record]
            com_ref_record_y = [record[1] for record in walking.com_traj_record]
            com_traj_record_x = [record[0] for record in com_traj_record]
            com_traj_record_y = [record[1] for record in com_traj_record]

            plot_line_graph(
                [
                    zmp_ref_record_y,
                    zmp_traj_record_y,
                    com_ref_record_y,
                    com_traj_record_y,
                ],
                x=[
                    zmp_ref_record_x,
                    zmp_traj_record_x,
                    com_ref_record_x,
                    com_traj_record_x,
                ],
                title="Footsteps Planning",
                x_label="X",
                y_label="Y",
                save_config=True,
                save_path="results/plots",
                time_suffix="",
                legend_labels=["ZMP Ref", "ZMP Traj", "CoM Ref", "Com Traj"],
                ax=ax,
                # checkpoint_period=[
                #     0,
                #     0,
                #     round(walking.fs_steps / 4),
                #     round(walking.fs_steps / 4),
                # ],
                checkpoint_period=[0, 0, 0, 0],
            )()

            plot_line_graph(
                joint_angle_errors,
                title="Joint Angle Errors",
                x_label="Simulation Step",
                y_label="Joint Angle Error",
                save_config=True,
                save_path="results/plots",
                time_suffix="",
            )()


if __name__ == "__main__":
    main()
