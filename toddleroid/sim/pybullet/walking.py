from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from toddleroid.control.preview_control import *
from toddleroid.planning.foot_step_planner import *
from toddleroid.sim.pybullet.robot import HumanoidRobot
from toddleroid.sim.pybullet.simulation import PyBulletSim
from toddleroid.utils.data_utils import round_floats


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(
        self,
        robot: HumanoidRobot,
        fsp: FootStepPlanner,
        pc: LQRPreviewController,
        left_foot0: List[float],
        right_foot0: List[float],
        joint_angles: List[float],
    ):
        """
        Initialize the walking parameters.

        Args:
            robot (HumanoidRobot): The robot instance.
            fsp (FootStepPlanner): The footstep planner.
            pc (PreviewControl): The preview control.
            left_foot0 (List[float]): Initial left foot position and orientation.
            right_foot0 (List[float]): Initial right foot position and orientation.
            joint_angles (List[float]): Initial joint angles.
        """
        self.robot = robot
        self.fsp = fsp
        self.pc = pc
        self.left_foot0, self.right_foot0 = left_foot0, right_foot0
        self.joint_angles = joint_angles

        self.control_matrix = np.zeros((3, 2))
        self.pattern = []
        self.left_up = self.right_up = 0.0
        self.left_offset, self.left_offset_goal, self.left_offset_delta = (
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
        self.right_offset, self.right_offset_goal, self.right_offset_delta = (
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
        self.theta = 0
        self.status = "start"
        self.next_support_leg = "right"
        self.foot_steps = []

    def set_goal_position(self, pos: Optional[np.ndarray] = None) -> List[FootStep]:
        """
        Set the goal position for the humanoid robot.

        Args:
            pos (Optional[Position]): The target position. If None, the robot starts or continues walking.

        Returns:
            List[FootStep]: A list of footsteps towards the goal position.
        """
        if pos is None:
            self._handle_no_position()
        else:
            self._update_foot_steps(pos)

        self._update_control_pattern()
        self._update_support_leg()

        # Update the theta value based on the current footstep.
        self.theta = self.foot_steps[0].position[2]

        return self.foot_steps

    def _handle_no_position(self):
        """Handle the case when no position is provided."""
        if len(self.foot_steps) <= 4:
            self.status = "start"
        if len(self.foot_steps) > 3:
            del self.foot_steps[0]

    def _update_foot_steps(self, pos: np.ndarray):
        """Update the foot steps based on the given position."""
        if len(self.foot_steps) > 2:
            if not self.status == "start":
                offset_y = -0.06 if self.next_support_leg == "left" else 0.06
            else:
                offset_y = 0.0

            current = np.array(
                [
                    self.foot_steps[1].position[0],
                    self.foot_steps[1].position[1] + offset_y,
                    self.foot_steps[1].position[2],
                ]
            )
        else:
            current = np.zeros(3)

        self.foot_steps = self.fsp.calculate_steps(
            pos, current, self.next_support_leg, self.status
        )
        self.status = "walking"

    def _update_control_pattern(self):
        """Update the control pattern based on the foot steps."""
        t = self.foot_steps[0].time
        self.pattern, com_state_curr = self.pc.compute_control_pattern(
            t, self.control_matrix[:, :2], self.foot_steps
        )
        self.control_matrix = com_state_curr.copy()

    def _update_support_leg(self):
        """Update the support leg and relevant offsets."""
        support_leg = self.foot_steps[0].support_leg
        next_step = self.foot_steps[1]
        offset_y = 0.06 if next_step.support_leg != "both" else 0.0
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
            self.right_offset_goal = offset
            self.right_offset_delta = (
                self.right_offset_goal - self.right_offset
            ) / 17.0
            self.next_support_leg = "right"
        elif support_leg == "right":
            self.left_offset_goal = offset
            self.left_offset_delta = (self.left_offset_goal - self.left_offset) / 17.0
            self.next_support_leg = "left"

    def get_next_position(self) -> Tuple[List[float], List[float], int]:
        """
        Calculate the next position of the robot based on the walking pattern.

        Returns:
            Tuple containing the next joint angles, left foot position, right foot position,
            pattern X position, and remaining pattern length.
        """
        pattern_first = self.pattern.pop(0)
        period = round((self.foot_steps[1].time - self.foot_steps[0].time) / 0.01)
        theta_change = (
            self.foot_steps[1].position[2] - self.foot_steps[0].position[2]
        ) / period
        self.theta += theta_change

        if self.foot_steps[0].support_leg == "right":
            self.left_up, self.left_offset = self._get_foot_offset(
                self.left_up,
                self.left_offset,
                self.left_offset_goal,
                self.left_offset_delta,
                period,
            )
        elif self.foot_steps[0].support_leg == "left":
            self.right_up, self.right_offset = self._get_foot_offset(
                self.right_up,
                self.right_offset,
                self.right_offset_goal,
                self.right_offset_delta,
                period,
            )

        left_foot_pos, left_foot_ori = self._get_foot_position(
            self.left_foot0, self.left_offset, self.left_up, pattern_first
        )
        right_foot_pos, right_foot_ori = self._get_foot_position(
            self.right_foot0, self.right_offset, self.right_up, pattern_first
        )
        self.joint_angles = self.robot.solve_ik(
            left_foot_pos,
            left_foot_ori,
            right_foot_pos,
            right_foot_ori,
            self.joint_angles,
        )
        xp = [pattern_first[2], pattern_first[3]]

        return self.joint_angles, xp, len(self.pattern)

    def _get_foot_offset(
        self,
        foot_up: float,
        foot_offset: np.ndarray,
        foot_offset_goal: np.ndarray,
        foot_offset_delta: np.ndarray,
        period: int,
    ) -> Tuple[float, np.array]:
        """
        Update the position of the specified foot during the walking cycle.

        Args:
            foot_up (float): Current vertical position of the foot.
            foot_offset (np.ndarray): Current offset of the foot.
            foot_offset_goal (np.ndarray): Final goal offset for the foot.
            foot_offset_delta (np.ndarray): Change in offset per step.
            period (int): Total period of the current walking cycle.
            foot_side (str): Side of the foot ('left' or 'right').

        Returns:
            Tuple[float, np.array]: Updated vertical position and offset of the foot.
        """
        BOTH_FOOT = round(0.17 / 0.01)
        start_up = round(BOTH_FOOT / 2)
        end_up = round(period / 2)
        period_up = end_up - start_up
        foot_height = 0.06

        # Determine the period range for foot movement
        period_range = period - len(self.pattern)

        # Up or down foot movement
        if start_up < period_range <= end_up:
            foot_up += foot_height / period_up
        elif foot_up > 0:
            foot_up = max(foot_up - foot_height / period_up, 0.0)

        # Move foot in the axes of x, y, theta
        if period_range > start_up:
            foot_offset += foot_offset_delta
            if period_range > (start_up + period_up * 2):
                foot_offset = foot_offset_goal.copy()

        return foot_up, foot_offset

    def _get_foot_position(
        self,
        foot_init: List[float],
        foot_offset: np.ndarray,
        foot_up: float,
        pattern_first: List[float],
    ) -> List[float]:
        """
        Calculate the position of the foot based on the current offsets and height.

        Args:
            foot_init (List[float]): Initial position and orientation of the foot.
            foot_offset (np.ndarray): Offset of the foot.
            foot_up (float): Height of the foot.
            pattern_first (List[float]): First pattern position.

        Returns:
            List[float]: Calculated position of the foot.
        """
        offset = foot_offset - np.block([[pattern_first[0:2], 0]])
        foot_x = foot_init[0] + offset[0, 0]
        foot_y = foot_init[1] + offset[0, 1]
        foot_z = foot_init[2] + foot_up
        foot_theta = self.theta - offset[0, 2]

        return [foot_x, foot_y, foot_z], [0.0, 0.0, foot_theta]


def main():
    import random

    random.seed(0)
    sim = PyBulletSim()
    # A 0.3725 offset moves the robot slightly up from the ground
    robot = HumanoidRobot("Sustaina_OP")
    sim.load_robot(robot)

    # TODO: Clean up the plan and control parameters
    planner_params = PlanParameters(
        max_stride=np.array([0.05, 0.03, 0.2]),
        period=0.34,
        width=0.06,
    )
    fsp = FootStepPlanner(planner_params)

    control_params = LQRPreviewControlParameters(
        com_height=0.3, dt=0.01, period=1.0, Q_val=1e8, R_val=1.0
    )

    pc = LQRPreviewController(control_params)

    link_name2idx = {p.getBodyInfo(robot.id)[0].decode("UTF-8"): -1}
    for idx in range(p.getNumJoints(robot.id)):
        link_name2idx[p.getJointInfo(robot.id, idx)[12].decode("UTF-8")] = idx

    left_foot = np.array(p.getLinkState(robot.id, link_name2idx["left_foot_link"])[0])
    right_foot = np.array(p.getLinkState(robot.id, link_name2idx["right_foot_link"])[0])

    left_offset_foot_to_sole = np.array([0.0, 0.01, -0.04])
    right_offset_foot_to_sole = np.array([0.0, -0.01, -0.04])

    left_sole = left_foot + left_offset_foot_to_sole
    right_sole = right_foot + right_offset_foot_to_sole

    joint_angles = []
    for idx in range(p.getNumJoints(robot.id)):
        if p.getJointInfo(robot.id, idx)[3] > -1:
            joint_angles += [0]

    walking = Walking(robot, fsp, pc, left_sole, right_sole, joint_angles)

    # goal position (x, y) theta
    foot_step = walking.set_goal_position(np.array([0.4, 0.0, 0.5]))
    j = 0
    while p.isConnected():
        j += 1

        if j >= 10:
            joint_angles, xp, n = walking.get_next_position()
            print(f"joint_angles: {round_floats(joint_angles[7:], 6)}")
            j = 0
            if n == 0:
                if len(foot_step) <= 5:
                    x_goal, y_goal, theta_goal = (
                        random.random() - 0.5,
                        random.random() - 0.5,
                        random.random() - 0.5,
                    )
                    print(f"Goal: ({x_goal}, {y_goal}, {theta_goal})")
                    foot_step = walking.set_goal_position(
                        np.array([x_goal, y_goal, theta_goal])
                    )
                else:
                    foot_step = walking.set_goal_position()

        for idx in range(p.getNumJoints(robot.id)):
            qIndex = p.getJointInfo(robot.id, idx)[3]
            if qIndex > -1:
                p.setJointMotorControl2(
                    robot.id, idx, p.POSITION_CONTROL, joint_angles[qIndex - 7]
                )

        p.stepSimulation()


if __name__ == "__main__":
    main()
