import argparse
import random
from typing import List, Optional, Tuple

import numpy as np

from toddleroid.control.lqr_preview import *
from toddleroid.planning.foot_step_planner import *
from toddleroid.sim.mujoco_sim import MujoCoSim
from toddleroid.sim.pybullet_sim import PyBulletSim
from toddleroid.sim.robot import HumanoidRobot
from toddleroid.tasks.walking_configs import *
from toddleroid.utils.data_utils import round_floats

random.seed(0)


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(
        self,
        robot: HumanoidRobot,
        config: WalkingConfig,
        left_sole_init: List[float],
        right_sole_init: List[float],
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
            max_stride=config.max_stride,
            period=config.plan_period,
            offset_y=config.y_offset_com_to_foot,
        )
        self.fsp = FootStepPlanner(plan_params)

        control_params = LQRPreviewControlParameters(
            com_height=robot.config.com_height,
            dt=config.control_dt,
            period=config.control_period,
            Q_val=config.control_cost_Q_val,
            R_val=config.control_cost_R_val,
        )
        self.pc = LQRPreviewController(control_params)

        self.left_sole_init, self.right_sole_init = left_sole_init, right_sole_init
        self.joint_angles = joint_angles

        self.com_traj = []
        self.com_state_curr = np.zeros((3, 2))

        self.foot_steps = []
        self.status = "start"
        self.next_support_leg = "right"

        self.control_steps = round(self.config.plan_period / self.config.control_dt)

        self.left_up = self.right_up = 0.0
        self.left_offset, self.left_offset_target, self.left_offset_delta = (
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
        self.right_offset, self.right_offset_target, self.right_offset_delta = (
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
        self.theta_curr = 0

    def plan_foot_steps(
        self,
        com_pos_target: Optional[np.ndarray] = None,
        com_state_fb: Optional[np.ndarray] = None,
    ) -> List[FootStep]:
        """
        Set the target position for the humanoid robot.

        Args:
            pos (Optional[np.ndarray]): The target position. If None, the robot starts or continues walking.

        Returns:
            List[FootStep]: A list of footsteps towards the target position.
        """
        if com_pos_target is None:
            self._handle_no_position()
        else:
            self._update_foot_steps(com_pos_target)

        if com_state_fb is not None:
            self.com_state_curr = com_state_fb

        # Update the com trajectory based on the foot steps.
        self.com_traj, self.com_state_curr = self.pc.compute_com_traj(
            self.com_state_curr, self.foot_steps
        )

        self._update_support_leg()

        # Update the theta value based on the current footstep.
        self.theta_curr = self.foot_steps[0].position[2]

        return self.foot_steps, self.com_traj

    def _handle_no_position(self):
        """Handle the case when no position is provided."""
        if len(self.foot_steps) <= 4:
            self.status = "start"

        if len(self.foot_steps) > 3:
            del self.foot_steps[0]

    def _update_foot_steps(self, com_pos_target: np.ndarray):
        """Update the foot steps based on the given position."""
        if len(self.foot_steps) > 2:
            if not self.status == "start":
                offset_y = (
                    -self.config.y_offset_com_to_foot
                    if self.next_support_leg == "left"
                    else self.config.y_offset_com_to_foot
                )
            else:
                offset_y = 0.0

            com_pos_curr = np.array(
                [
                    self.foot_steps[1].position[0],
                    self.foot_steps[1].position[1] + offset_y,
                    self.foot_steps[1].position[2],
                ]
            )
        else:
            com_pos_curr = np.zeros(3)

        self.foot_steps = self.fsp.compute_steps(
            com_pos_target, com_pos_curr, self.next_support_leg, self.status
        )
        self.status = "walking"

    def _update_support_leg(self):
        """Update the support leg and relevant offsets."""
        support_leg = self.foot_steps[0].support_leg
        next_step = self.foot_steps[1]
        offset_y = (
            self.config.y_offset_com_to_foot if next_step.support_leg != "both" else 0.0
        )
        offset = np.array(
            [
                [
                    next_step.position[0],
                    next_step.position[1]
                    + (offset_y if support_leg == "left" else -offset_y),
                    next_step.position[2],
                ]
            ]
        )

        if support_leg == "left":
            self.right_offset_target = offset
            self.right_offset_delta = (self.right_offset_target - self.right_offset) / (
                self.control_steps / 2
            )
            self.next_support_leg = "right"
        elif support_leg == "right":
            self.left_offset_target = offset
            self.left_offset_delta = (self.left_offset_target - self.left_offset) / (
                self.control_steps / 2
            )
            self.next_support_leg = "left"

    def solve_joint_angles(self) -> Tuple[List[float], List[float], int]:
        """
        Calculate the next position of the robot based on the walking pattern.

        Returns:
            Tuple containing the next joint angles, left foot position, right foot position,
            pattern X position, and remaining pattern length.
        """
        com_pos = self.com_traj.pop(0)
        theta_change = (
            self.foot_steps[1].position[2] - self.foot_steps[0].position[2]
        ) / self.control_steps
        self.theta_curr += theta_change

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
            self.left_sole_init, self.left_offset, self.left_up, com_pos
        )
        right_foot_pos, right_foot_ori = self._get_foot_position(
            self.right_sole_init, self.right_offset, self.right_up, com_pos
        )
        self.joint_angles = self.robot.solve_ik(
            left_foot_pos,
            left_foot_ori,
            right_foot_pos,
            right_foot_ori,
            self.joint_angles,
        )
        is_traj_completed = len(self.com_traj) == 0

        return self.joint_angles, is_traj_completed

    def _get_foot_offset(
        self,
        foot_up: float,
        foot_offset: np.ndarray,
        foot_offset_target: np.ndarray,
        foot_offset_delta: np.ndarray,
    ) -> Tuple[float, np.array]:
        """
        Update the position of the specified foot during the walking cycle.

        Args:
            foot_up (float): Current vertical position of the foot.
            foot_offset (np.ndarray): Current offset of the foot.
            foot_offset_target (np.ndarray): Final target offset for the foot.
            foot_offset_delta (np.ndarray): Change in offset per step.
            control_steps (int): Total control_steps of the current walking cycle.
            foot_side (str): Side of the foot ('left' or 'right').

        Returns:
            Tuple[float, np.array]: Updated vertical position and offset of the foot.
        """
        move_up_step_start = round(self.control_steps / 4)
        move_up_step_end = round(self.control_steps / 2)
        move_up_period = move_up_step_end - move_up_step_start

        # Determine the period range for foot movement
        control_steps_left_curr = self.control_steps - len(self.com_traj)

        # Up or down foot movement
        foot_up_delta = self.config.foot_step_height / move_up_period
        if move_up_step_start < control_steps_left_curr <= move_up_step_end:
            foot_up += foot_up_delta
        else:
            foot_up = max(foot_up - foot_up_delta, 0.0)

        # Move foot in the axes of x, y, theta
        if control_steps_left_curr > move_up_step_start:
            foot_offset += foot_offset_delta
            if control_steps_left_curr > (move_up_step_start + move_up_period * 2):
                foot_offset = foot_offset_target.copy()

        return foot_up, foot_offset

    def _get_foot_position(
        self,
        foot_init: List[float],
        foot_offset: np.ndarray,
        foot_up: float,
        com_pos: List[float],
    ) -> List[float]:
        """
        Calculate the position of the foot based on the current offsets and height.

        Args:
            foot_init (List[float]): Initial position and orientation of the foot.
            foot_offset (np.ndarray): Offset of the foot.
            foot_up (float): Height of the foot.
            com_pos (List[float]): First pattern position.

        Returns:
            List[float]: Calculated position of the foot.
        """
        offset = foot_offset - np.concatenate([com_pos, [0.0]])
        foot_x = foot_init[0] + offset[0, 0]
        foot_y = foot_init[1] + offset[0, 1]
        foot_z = foot_init[2] + foot_up
        foot_theta = self.theta_curr - offset[0, 2]

        return [foot_x, foot_y, foot_z], [0.0, 0.0, foot_theta]


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

    left_foot_link_pos = sim.get_link_pos(
        robot, robot.config.canonical_name2link_name["left_foot_link"]
    )
    right_foot_link_pos = sim.get_link_pos(
        robot, robot.config.canonical_name2link_name["right_foot_link"]
    )

    left_foot_pos_init = (
        left_foot_link_pos + robot.config.offsets["left_offset_foot_to_sole"]
    )
    right_foot_pos_init = (
        right_foot_link_pos + robot.config.offsets["right_offset_foot_to_sole"]
    )

    joint_angles = sim.initialize_joint_angles(robot)
    if robot.name == "robotis_op3":
        joint_angles["l_sho_roll"] = np.pi / 4
        joint_angles["r_sho_roll"] = -np.pi / 4

    walking = Walking(
        robot, config, left_foot_pos_init, right_foot_pos_init, joint_angles
    )

    sim_step_idx = 0
    foot_steps, com_traj = walking.plan_foot_steps(config.target_pos_init)

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx, foot_steps, com_traj, joint_angles):
        sim_step_idx += 1
        if sim_step_idx >= config.sim_step_interval:
            sim_step_idx = 0
            joint_angles, is_traj_completed = walking.solve_joint_angles()
            print(f"joint_angles: {round_floats(list(joint_angles.values()), 6)}")

            if is_traj_completed:
                if len(foot_steps) <= 5:
                    target_x, target_y, theta_target = (
                        random.random() - 0.5,
                        random.random() - 0.5,
                        random.random() - 0.5,
                    )
                    print(f"Goal: ({target_x}, {target_y}, {theta_target})")
                    com_pos_target = np.array([target_x, target_y, theta_target])
                else:
                    com_pos_target = None

                if args.use_feedback:
                    com_state_fb = sim.get_com_state(robot)
                else:
                    com_state_fb = None

                foot_steps, com_traj = walking.plan_foot_steps(
                    com_pos_target, com_state_fb
                )

        sim.set_joint_angles(robot, joint_angles)
        return sim_step_idx, foot_steps, com_traj, joint_angles

    sim.simulate(
        step_func,
        (sim_step_idx, foot_steps, com_traj, joint_angles),
        args.sleep_time,
        vis_flags=["foot_steps", "com_traj"],
    )


if __name__ == "__main__":
    main()
