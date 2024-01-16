import random
from time import sleep

import numpy as np
import pybullet as p

from toddleroid.control.preview_control import *
from toddleroid.planning.foot_step_planner import *
from toddleroid.sim.pybullet.robot import HumanoidRobot
from toddleroid.sim.pybullet.simulation import PyBulletSim
from toddleroid.utils.data_utils import round_floats


class Walking:
    def __init__(self, robot, fsp, pc, left_foot0, right_foot0, joint_angles):
        self.robot = robot
        self.fsp = fsp
        self.pc = pc
        self.left_foot0, self.right_foot0 = left_foot0, right_foot0
        self.joint_angles = joint_angles

        self.X = np.matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.pattern = []
        self.left_up = self.right_up = 0.0
        self.left_off, self.left_off_g, self.left_off_d = (
            np.matrix([[0.0, 0.0, 0.0]]),
            np.matrix([[0.0, 0.0, 0.0]]),
            np.matrix([[0.0, 0.0, 0.0]]),
        )
        self.right_off, self.right_off_g, self.right_off_d = (
            np.matrix([[0.0, 0.0, 0.0]]),
            np.matrix([[0.0, 0.0, 0.0]]),
            np.matrix([[0.0, 0.0, 0.0]]),
        )
        self.th = 0
        self.status = "start"
        self.next_support_leg = "right"
        self.foot_step = []

    def setGoalPos(self, pos: Position = None):
        if pos == None:
            if len(self.foot_step) <= 4:
                self.status = "start"
            if len(self.foot_step) > 3:
                del self.foot_step[0]
        else:
            if len(self.foot_step) > 2:
                if not self.status == "start":
                    offset_y = -0.06 if self.next_support_leg == "left" else 0.06
                else:
                    offset_y = 0.0

                current = Position(
                    x=self.foot_step[1].position.x,
                    y=self.foot_step[1].position.y + offset_y,
                    theta=self.foot_step[1].position.theta,
                )
            else:
                current = Position(x=0.0, y=0.0, theta=0.0)

            self.foot_step = self.fsp.calculate_steps(
                pos,
                current,
                self.next_support_leg,
                self.status,
            )
            self.status = "walking"

        t = self.foot_step[0].time
        self.pattern, x, y = self.pc.compute_control_pattern(
            t, self.X[:, 0], self.X[:, 1], self.foot_step
        )
        self.X = np.matrix([[x[0, 0], y[0, 0]], [x[1, 0], y[1, 0]], [x[2, 0], y[2, 0]]])
        if self.foot_step[0].support_leg == "left":
            if self.foot_step[1].support_leg == "both":
                self.right_off_g = np.matrix(
                    [
                        [
                            self.foot_step[1].position.x,
                            self.foot_step[1].position.y,
                            self.foot_step[1].position.theta,
                        ]
                    ]
                )
            else:
                self.right_off_g = np.matrix(
                    [
                        [
                            self.foot_step[1].position.x,
                            self.foot_step[1].position.y + 0.06,
                            self.foot_step[1].position.theta,
                        ]
                    ]
                )
            self.right_off_d = (self.right_off_g - self.right_off) / 17.0
            self.next_support_leg = "right"
        if self.foot_step[0].support_leg == "right":
            if self.foot_step[1].support_leg == "both":
                self.left_off_g = np.matrix(
                    [
                        [
                            self.foot_step[1].position.x,
                            self.foot_step[1].position.y,
                            self.foot_step[1].position.theta,
                        ]
                    ]
                )
            else:
                self.left_off_g = np.matrix(
                    [
                        [
                            self.foot_step[1].position.x,
                            self.foot_step[1].position.y - 0.06,
                            self.foot_step[1].position.theta,
                        ]
                    ]
                )
            self.left_off_d = (self.left_off_g - self.left_off) / 17.0
            self.next_support_leg = "left"

        self.th = self.foot_step[0].position.theta

        return self.foot_step

    def getNextPos(self):
        X = self.pattern.pop(0)
        period = round((self.foot_step[1].time - self.foot_step[0].time) / 0.01)
        self.th += (
            self.foot_step[1].position.theta - self.foot_step[0].position.theta
        ) / period
        BOTH_FOOT = round(0.17 / 0.01)
        start_up = round(BOTH_FOOT / 2)
        end_up = round(period / 2)
        period_up = end_up - start_up
        foot_hight = 0.06
        if self.foot_step[0].support_leg == "right":
            # up or down foot
            if start_up < (period - len(self.pattern)) <= end_up:
                self.left_up += foot_hight / period_up
            elif self.left_up > 0:
                self.left_up = max(self.left_up - foot_hight / period_up, 0.0)
            # move foot in the axes of x,y,the
            if (period - len(self.pattern)) > start_up:
                self.left_off += self.left_off_d
                if (period - len(self.pattern)) > (start_up + period_up * 2):
                    self.left_off = self.left_off_g.copy()
        if self.foot_step[0].support_leg == "left":
            # up or down foot
            if start_up < (period - len(self.pattern)) <= end_up:
                self.right_up += foot_hight / period_up
            elif self.right_up > 0:
                self.right_up = max(self.right_up - foot_hight / period_up, 0.0)
            # move foot in the axes of x,y,the
            if (period - len(self.pattern)) > start_up:
                self.right_off += self.right_off_d
                if (period - len(self.pattern)) > (start_up + period_up * 2):
                    self.right_off = self.right_off_g.copy()
        lo = self.left_off - np.block([[X[0:2], 0]])
        ro = self.right_off - np.block([[X[0:2], 0]])
        left_foot = [
            self.left_foot0[0] + lo[0, 0],
            self.left_foot0[1] + lo[0, 1],
            self.left_foot0[2] + self.left_up,
            0.0,
            0.0,
            self.th - lo[0, 2],
        ]
        right_foot = [
            self.right_foot0[0] + ro[0, 0],
            self.right_foot0[1] + ro[0, 1],
            self.right_foot0[2] + self.right_up,
            0.0,
            0.0,
            self.th - ro[0, 2],
        ]
        self.joint_angles = self.robot.solve_ik(
            left_foot[:3],
            left_foot[3:],
            right_foot[:3],
            right_foot[3:],
            self.joint_angles,
        )
        xp = [X[2], X[3]]

        return self.joint_angles, left_foot, right_foot, xp, len(self.pattern)


def main():
    random.seed(0)
    sim = PyBulletSim()
    robot = HumanoidRobot("Sustaina_OP")
    sim.load_robot(robot)

    index = {p.getBodyInfo(robot.id)[0].decode("UTF-8"): -1}
    for id in range(p.getNumJoints(robot.id)):
        index[p.getJointInfo(robot.id, id)[12].decode("UTF-8")] = id

    left_foot0 = p.getLinkState(robot.id, index["left_foot_link"])[0]
    right_foot0 = p.getLinkState(robot.id, index["right_foot_link"])[0]

    joint_angles = []
    for id in range(p.getNumJoints(robot.id)):
        if p.getJointInfo(robot.id, id)[3] > -1:
            joint_angles += [0]

    left_foot = [left_foot0[0] - 0.0, left_foot0[1] + 0.01, left_foot0[2] - 0.04]
    right_foot = [right_foot0[0] - 0.0, right_foot0[1] - 0.01, right_foot0[2] - 0.04]

    planner_params = PlanParameters(
        max_stride_x=0.05,
        max_stride_y=0.03,
        max_stride_th=0.2,
        period=0.34,
        width=0.06,
    )

    fsp = FootStepPlanner(planner_params)

    control_params = ControlParameters(dt=0.01, period=1.0, Q_val=1e8, H_val=1.0)

    robot_height = 0.30
    # State-space matrices for preview control
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0], [0], [1]])
    C = np.array([[1, 0, -robot_height / GRAVITY]])
    D = np.array([[0]])
    state_space = StateSpace(A, B, C, D)

    pc = PreviewControl(control_params, state_space)

    walking = Walking(robot, fsp, pc, left_foot, right_foot, joint_angles)

    index_dof = {
        p.getBodyInfo(robot.id)[0].decode("UTF-8"): -1,
    }
    for id in range(p.getNumJoints(robot.id)):
        index_dof[p.getJointInfo(robot.id, id)[12].decode("UTF-8")] = (
            p.getJointInfo(robot.id, id)[3] - 7
        )

    # goal position (x, y) theta
    foot_step = walking.setGoalPos(Position(x=0.4, y=0.0, theta=0.5))
    j = 0
    while p.isConnected():
        j += 1

        if j >= 10:
            joint_angles, lf, rf, xp, n = walking.getNextPos()
            print(f"joint_angles: {round_floats(joint_angles[7:], 6)}")
            j = 0
            if n == 0:
                if len(foot_step) <= 5:
                    x_goal, y_goal, th = (
                        random.random() - 0.5,
                        random.random() - 0.5,
                        random.random() - 0.5,
                    )
                    # x_goal, y_goal, th = 0.2, 0.2, 0.2
                    print(f"Goal: ({x_goal}, {y_goal}, {th})")
                    foot_step = walking.setGoalPos(
                        Position(x=x_goal, y=y_goal, theta=th)
                    )
                else:
                    foot_step = walking.setGoalPos()

        for id in range(p.getNumJoints(robot.id)):
            qIndex = p.getJointInfo(robot.id, id)[3]
            if qIndex > -1:
                p.setJointMotorControl2(
                    robot.id, id, p.POSITION_CONTROL, joint_angles[qIndex - 7]
                )

        p.stepSimulation()


if __name__ == "__main__":
    main()
