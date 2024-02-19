import argparse
import random
from typing import List, Optional, Tuple

import numpy as np

from toddlerbot.control.zmp_preview_control import *
from toddlerbot.planning.foot_step_planner import *
from toddlerbot.sim.mujoco_sim import MujoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.tasks.walking_configs import *
from toddlerbot.utils.data_utils import round_floats
from toddlerbot.utils.vis_plot import *

random.seed(0)


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(
        self,
        robot: HumanoidRobot,
        config: WalkingConfig,
        com_pos_curr: np.ndarray,
        joint_angles: List[float],
    ):
        """
        Initialize the walking parameters.

        Args:
            robot (HumanoidRobot): The robot instance.
            config (WalkingConfig): The walking configuration.
            left_sole_init (List[float]): Initial left foot position and orientation.
            right_sole_init (List[float]): Initial right foot position and orientation.
            joint_angles (List[float]): Initial joint angles.
        """
        self.robot = robot
        self.config = config

        plan_params = FootStepPlanParameters(
            max_stride=config.plan_max_stride,
            t_step=config.plan_t_step,
            y_offset_zmp=self.config.y_offset_zmp,
        )
        self.fsp = FootStepPlanner(plan_params)

        control_params = ZMPPreviewControlParameters(
            com_z=robot.config.com_z,
            dt=config.control_dt,
            t_preview=config.control_t_preview,
            Q_val=config.control_cost_Q_val,
            R_val=config.control_cost_R_val,
        )
        self.pc = ZMPPreviewController(control_params)

        # self.left_ank_pos_init, self.right_ank_pos_init = (
        #     left_ank_pos_init,
        #     right_ank_pos_init,
        # )
        self.joint_angles = joint_angles
        self.idx = 0

        self.zmp_ref_record = []
        self.zmp_traj_record = []
        self.com_traj_record = []

        self.com_traj_queue = []
        self.com_curr = np.zeros((3, 2))
        # np.concatenate([com_pos_curr[None, :2], np.zeros((2, 2))], axis=0)

        self.foot_steps = []
        self.status = "start"
        self.next_support_leg = "right"

        self.fs_steps = round(self.config.plan_t_step / self.config.control_dt)

        self.left_up = self.right_up = 0.0
        self.left_offset, self.left_offset_target, self.left_offset_delta = (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )
        self.right_offset, self.right_offset_target, self.right_offset_delta = (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )
        self.theta_curr = 0

    def plan_foot_steps(
        self,
        target_pose: Optional[np.ndarray] = None,
        zmp_fb: Optional[np.ndarray] = None,
    ) -> List[FootStep]:
        if target_pose is None:
            if len(self.foot_steps) <= 4:
                self.status = "start"

            if len(self.foot_steps) > 3:
                del self.foot_steps[0]
        else:
            self._update_foot_steps(target_pose)

        # if zmp_fb is not None:
        #     self.com_curr = zmp_fb

        # Update the com trajectory based on the foot steps.
        zmp_ref, zmp_traj, com_traj, self.com_curr = self.pc.compute_com_traj(
            self.com_curr, self.foot_steps
        )
        self.com_traj_queue = com_traj

        self.zmp_ref_record.extend(zmp_ref)
        self.zmp_traj_record.extend(zmp_traj)
        self.com_traj_record.extend(com_traj)

        # Update the theta value based on the current footstep.
        self.theta_curr = self.foot_steps[0].position[2]

        return self.foot_steps, self.com_traj_queue

    def _update_foot_steps(self, target_pose: np.ndarray):
        foot_step_curr = np.zeros(3)
        path, self.foot_steps = self.fsp.compute_steps(foot_step_curr, target_pose)
        self.status = "walking"

    def _update_support_leg(self):
        """Update the support leg and relevant offsets."""
        support_leg = self.foot_steps[0].support_leg
        next_step = self.foot_steps[1]
        offset_y = 0.0 if next_step.support_leg == "both" else self.config.y_offset_zmp
        offset_target = np.array(
            [
                next_step.position[0],
                next_step.position[1]
                + (offset_y if support_leg == "left" else -offset_y),
                next_step.position[2],
            ]
        )
        if support_leg == "left":
            self.right_offset_target = offset_target
            self.right_offset_delta = (self.right_offset_target - self.right_offset) / (
                self.fs_steps / 2
            )
            self.next_support_leg = "right"
        elif support_leg == "right":
            self.left_offset_target = offset_target
            self.left_offset_delta = (self.left_offset_target - self.left_offset) / (
                self.fs_steps / 2
            )
            self.next_support_leg = "left"

    def solve_joint_angles(self) -> Tuple[List[float], List[float], int]:
        if len(self.com_traj_queue) == 0:
            return self.joint_angles

        theta_change = (
            self.foot_steps[1].position[2] - self.foot_steps[0].position[2]
        ) / self.fs_steps
        self.theta_curr += theta_change

        com_pos = self.com_traj_queue.pop(0)

        if abs(self.idx * self.config.control_dt - self.foot_steps[1].time) < 1e-6:
            self.foot_steps.pop(0)
            self._update_support_leg()

        if self.foot_steps[0].support_leg == "right":
            self.left_up, self.left_offset = self._get_foot_offset(
                self.left_up,
                self.left_offset,
                self.left_offset_target,
                self.left_offset_delta,
            )
        elif self.foot_steps[0].support_leg == "left":
            self.right_up, self.right_offset = self._get_foot_offset(
                self.right_up,
                self.right_offset,
                self.right_offset_target,
                self.right_offset_delta,
            )

        left_foot_pos, left_foot_ori = self._get_foot_position(
            com_pos, self.left_offset, self.left_up
        )
        right_foot_pos, right_foot_ori = self._get_foot_position(
            com_pos, self.right_offset, self.right_up
        )

        self.joint_angles = self.robot.solve_ik(
            left_foot_pos,
            left_foot_ori,
            right_foot_pos,
            right_foot_ori,
            self.joint_angles,
        )
        self.idx += 1

        return self.joint_angles

    def _get_foot_offset(
        self,
        foot_up: float,
        foot_offset: np.ndarray,
        foot_offset_target: np.ndarray,
        foot_offset_delta: np.ndarray,
    ) -> Tuple[float, np.array]:
        up_start_idx = round(self.fs_steps / 4)
        up_end_idx = round(self.fs_steps / 2)
        up_period = up_end_idx - up_start_idx

        idx_curr = self.idx % self.fs_steps

        # Up or down foot movement
        foot_up_delta = self.config.foot_step_height / up_period
        if up_start_idx < idx_curr <= up_end_idx:
            foot_up += foot_up_delta
        else:
            foot_up = max(foot_up - foot_up_delta, 0.0)

        # Move foot in the axes of x, y, theta
        if idx_curr > up_start_idx:
            foot_offset += foot_offset_delta
            if idx_curr > (up_start_idx + up_period * 2):
                foot_offset = foot_offset_target.copy()

        return foot_up, foot_offset

    def _get_foot_position(
        self,
        com_pos: List[float],
        foot_offset: np.ndarray,
        foot_up: float,
    ) -> List[float]:
        offset = (foot_offset - np.concatenate([com_pos, [0.0]])).squeeze()
        foot_pos = [*offset[:2], self.config.squat_height + foot_up]
        foot_theta = self.theta_curr - offset[2]

        return foot_pos, [0.0, 0.0, foot_theta]


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

    # A 0.3725 offset moves the robot slightly up from the ground
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

    com_pos_curr = sim.get_com(robot)
    walking = Walking(robot, config, com_pos_curr, joint_angles)

    joint_angle_errors = []

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx, foot_steps, com_traj_queue, joint_angles):
        nonlocal joint_angle_errors

        sim_step_idx += 1
        if args.sim == "mujoco":
            joint_angles_error = sim.get_joint_angles_error(robot, joint_angles)
            joint_angle_errors.append(np.linalg.norm(list(joint_angles_error.values())))

        if sim_step_idx >= config.actuator_steps:
            sim_step_idx = 0
            joint_angles = walking.solve_joint_angles()
            print(f"joint_angles: {round_floats(list(joint_angles.values()), 6)}")

            # if is_finished:
            #     if len(foot_steps) <= 5:
            #         target_x, target_y, theta_target = (
            #             random.random() - 0.5,
            #             random.random() - 0.5,
            #             0,  # random.random() - 0.5,
            #         )
            #         print(f"Goal: ({target_x}, {target_y}, {theta_target})")
            #         target_pose = np.array([target_x, target_y, theta_target])
            #     else:
            #         target_pose = None

            #     # if args.use_feedback:
            #     #     zmp_fb = sim.get_zmp(robot)
            #     # else:
            #     #     zmp_fb = None

            #     foot_steps, com_traj_queue = walking.plan_foot_steps(target_pose)

        sim.set_joint_angles(robot, joint_angles)

        return sim_step_idx, foot_steps, com_traj_queue, joint_angles

    # time_suffix = "_" + time.strftime("%Y%m%d_%H%M%S")
    sim_step_idx = 0
    foot_steps, com_traj_queue = walking.plan_foot_steps(config.target_pose_init)
    try:
        sim.simulate(
            step_func,
            (sim_step_idx, foot_steps, com_traj_queue, joint_angles),
            args.sleep_time,
            vis_flags=["foot_steps", "com_traj_queue"],
        )
    finally:
        if args.sim == "mujoco":
            plot_line_graph(
                joint_angle_errors,
                title="Joint Angle Errors",
                x_label="Simulation Step",
                y_label="Joint Angle Error",
                save_config=True,
                save_path="results/plots",
                time_suffix="",
            )()

            zmp_ref_record_x = [record[0] for record in walking.zmp_ref_record]
            zmp_ref_record_y = [record[1] for record in walking.zmp_ref_record]
            zmp_traj_record_x = [record[0] for record in walking.zmp_traj_record]
            zmp_traj_record_y = [record[1] for record in walking.zmp_traj_record]
            com_traj_record_x = [record[0] for record in walking.com_traj_record]
            com_traj_record_y = [record[1] for record in walking.com_traj_record]

            plot_line_graph(
                [zmp_ref_record_y, zmp_traj_record_y, com_traj_record_y],
                x=[zmp_ref_record_x, zmp_traj_record_x, com_traj_record_x],
                title="ZMP Reference",
                x_label="X",
                y_label="Y",
                save_config=True,
                save_path="results/plots",
                time_suffix="",
                legend_labels=["ZMP Ref", "ZMP Traj", "CoM Traj"],
            )()


if __name__ == "__main__":
    main()
