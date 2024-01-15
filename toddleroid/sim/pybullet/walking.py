import csv

import numpy as np
import pybullet as p

from toddleroid.control.preview_control import ControlParameters, PreviewControl
from toddleroid.planning.foot_step_planner import (
    FootStepPlanner,
    PlanParameters,
    Position,
    Step,
)
from toddleroid.sim.pybullet.kinematics import Kinematics
from toddleroid.sim.pybullet.simulation import PyBulletSim
from toddleroid.sim.robot import HumanoidRobot


class Walking:
    def __init__(self, robot_id, left_foot0, right_foot0, joint_angles):
        self.kine = Kinematics(robot_id)
        self.left_foot0, self.right_foot0 = left_foot0, right_foot0
        self.joint_angles = joint_angles

        # Initialize control parameters and state space for Preview Control
        control_params = ControlParameters(dt=0.01, period=1.0, Q_val=1e8, H_val=1.0)
        # You need to define 'state_space' based on your robot's state-space representation
        state_space = None  # Placeholder, replace with actual StateSpace instance
        self.pc = PreviewControl(control_params, state_space)

        plan_params = PlanParameters(
            max_stride_x=0.05,
            max_stride_y=0.03,
            max_stride_th=0.2,
            period=0.34,
            width=0.06,
        )
        self.fsp = FootStepPlanner(plan_params)

        # Initialize other necessary attributes
        self.pattern = []
        self.left_up = self.right_up = 0.0
        self.left_off = self.right_off = np.matrix([[0.0, 0.0, 0.0]])
        self.th = 0
        self.status = "start"
        self.next_leg = "right"
        self.foot_step = []

    def set_goal_position(self, pos=None):
        if pos is None:
            self._update_status_and_steps_for_none_position()
        else:
            self._update_status_and_steps_for_given_position(pos)

        # Update the pattern, state matrix X, and foot offset calculations
        self._update_pattern_and_state_matrix()
        self._update_foot_offsets()

        self.th = self.foot_step[0].theta
        return self.foot_step

    def _update_status_and_steps_for_none_position(self):
        if len(self.foot_step) <= 4:
            self.status = "start"
        if len(self.foot_step) > 3:
            del self.foot_step[0]

    def _update_status_and_steps_for_given_position(self, pos):
        current_x, current_y, current_th = self._get_current_foot_position()
        self.foot_step = self.fsp.calculate_steps(
            pos, Position(current_x, current_y, current_th), self.next_leg, self.status
        )
        self.status = "walking"

    def _get_current_foot_position(self):
        if len(self.foot_step) > 2 and self.status != "start":
            offset_y = -0.06 if self.next_leg == "left" else 0.06
        else:
            offset_y = 0.0
        current_step = self.foot_step[1]
        return current_step.x, current_step.y + offset_y, current_step.theta

    def _update_pattern_and_state_matrix(self):
        t = self.foot_step[0].time
        self.pattern, x, y = self.pc.set_param(
            t, self.X[:, 0], self.X[:, 1], self.foot_step
        )
        self.X = np.matrix([[x[0, 0], y[0, 0]], [x[1, 0], y[1, 0]], [x[2, 0], y[2, 0]]])

    def _update_foot_offsets(self):
        if self.foot_step[0].support_leg == "left":
            self._calculate_right_foot_offset()
            self.next_leg = "right"
        elif self.foot_step[0].support_leg == "right":
            self._calculate_left_foot_offset()
            self.next_leg = "left"

    def _calculate_right_foot_offset(self):
        goal_step = self.foot_step[1]
        offset_y = 0.06 if goal_step.support_leg != "both" else 0.0
        self.right_off_g = np.matrix(
            [[goal_step.x, goal_step.y + offset_y, goal_step.theta]]
        )
        self.right_off_d = (self.right_off_g - self.right_off) / 17.0

    def _calculate_left_foot_offset(self):
        goal_step = self.foot_step[1]
        offset_y = -0.06 if goal_step.support_leg != "both" else 0.0
        self.left_off_g = np.matrix(
            [[goal_step.x, goal_step.y + offset_y, goal_step.theta]]
        )
        self.left_off_d = (self.left_off_g - self.left_off) / 17.0

    def get_next_position(self):
        next_pattern_element = self.pattern.pop(0)
        period = self._calculate_period()
        self.th += self._calculate_theta_increment(period)

        self._update_foot_vertical_movement(period, next_pattern_element)
        self._update_foot_horizontal_movement(period)

        left_foot_pos = self._calculate_foot_position(
            self.left_foot0, self.left_off, self.left_up
        )
        right_foot_pos = self._calculate_foot_position(
            self.right_foot0, self.right_off, self.right_up
        )

        self.joint_angles = self.kine.solve_ik(
            left_foot_pos, right_foot_pos, self.joint_angles
        )
        xp = self._get_xp(next_pattern_element)

        return self.joint_angles, left_foot_pos, right_foot_pos, xp, len(self.pattern)

    def _calculate_period(self):
        return round((self.foot_step[1].time - self.foot_step[0].time) / 0.01)

    def _calculate_theta_increment(self, period):
        return (self.foot_step[1].theta - self.foot_step[0].theta) / period

    def _update_foot_vertical_movement(self, period, next_pattern_element):
        start_up, end_up, period_up, foot_height = self._get_vertical_movement_params(
            period
        )
        if self._is_updating_vertical_position(period, start_up, end_up):
            increment = foot_height / period_up
            if self.foot_step[0].support_leg == "right":
                self.left_up += increment
            elif self.foot_step[0].support_leg == "left":
                self.right_up += increment

        self.left_up = self._decrease_if_positive(self.left_up, increment)
        self.right_up = self._decrease_if_positive(self.right_up, increment)

    def _update_foot_horizontal_movement(self, period):
        start_up, end_up, period_up, _ = self._get_vertical_movement_params(period)
        if period - len(self.pattern) > start_up:
            if self.foot_step[0].support_leg == "right":
                self.left_off += self.left_off_d
            elif self.foot_step[0].support_leg == "left":
                self.right_off += self.right_off_d

        # Final position adjustments
        if period - len(self.pattern) > (start_up + period_up * 2):
            if self.foot_step[0].support_leg == "right":
                self.left_off = self.left_off_g.copy()
            elif self.foot_step[0].support_leg == "left":
                self.right_off = self.right_off_g.copy()

    def _calculate_foot_position(self, initial_foot_pos, offset, vertical_movement):
        foot_position = [
            initial_foot_pos[0] + offset[0, 0],
            initial_foot_pos[1] + offset[0, 1],
            initial_foot_pos[2] + vertical_movement,
            0.0,  # Assuming constant orientation (pitch and roll)
            0.0,  # Assuming constant orientation (pitch and roll)
            self.th - offset[0, 2],  # Adjust theta (yaw)
        ]
        return foot_position

    def _get_vertical_movement_params(self, period):
        both_foot_duration = round(0.17 / 0.01)
        start_up = round(both_foot_duration / 2)
        end_up = round(period / 2)
        period_up = end_up - start_up
        foot_height = 0.06
        return start_up, end_up, period_up, foot_height

    def _is_updating_vertical_position(self, period, start_up, end_up):
        return start_up < (period - len(self.pattern)) <= end_up

    def _decrease_if_positive(self, value, decrement):
        return max(value - decrement, 0.0) if value > 0 else value

    def _get_xp(self, pattern_element):
        return [pattern_element[0, 2], pattern_element[0, 3]]


def main():
    sim = PyBulletSim(timestep=0.001)
    robot = HumanoidRobot("gankenkun")  # Replace with actual robot name
    sim.load_robot(robot)

    # Initialize joint angles and foot positions
    joint_angles = [0] * p.getNumJoints(robot.id)
    left_foot0, right_foot0 = get_initial_foot_positions(robot.id)

    # Initialize the walking class
    pc = PreviewControl(
        0.01, 1.0, 0.27
    )  # Replace with actual preview control initialization
    walk = Walking(robot.id, left_foot0, right_foot0, joint_angles, pc)

    # Set initial goal position
    walk.set_goal_position([0.4, 0.0, 0.0])

    # Simulation loop
    j = 0
    foot_step = []
    with open("result.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        while p.isConnected():
            j += 1
            if j >= 10:
                joint_angles, lf, rf, xp, n = walk.get_next_position()
                writer.writerow(np.concatenate([lf, rf, xp]))
                j = 0
                update_foot_step(walk, foot_step)
            update_robot_joints(robot.id, joint_angles)
            p.stepSimulation()


def get_initial_foot_positions(robot_id):
    left_foot_link_id = p.getBodyInfo(robot_id)[0].decode("UTF-8")
    right_foot_link_id = p.getBodyInfo(robot_id)[1].decode(
        "UTF-8"
    )  # Adjust indices as needed
    left_foot0 = p.getLinkState(robot_id, left_foot_link_id)[0]
    right_foot0 = p.getLinkState(robot_id, right_foot_link_id)[0]
    return left_foot0, right_foot0


def update_foot_step(walk, foot_step):
    if len(foot_step) <= 6:
        new_pos = [
            foot_step[-1][1] + 0.4,
            foot_step[-1][2] + 0.1,
            foot_step[-1][3] + 0.5,
        ]
        foot_step = walk.set_goal_position(new_pos)
    else:
        foot_step = walk.set_goal_position()


def update_robot_joints(robot_id, joint_angles):
    for id in range(p.getNumJoints(robot_id)):
        qIndex = p.getJointInfo(robot_id, id)[3]
        if qIndex > -1:
            p.setJointMotorControl2(
                robot_id, id, p.POSITION_CONTROL, joint_angles[qIndex - 7]
            )
