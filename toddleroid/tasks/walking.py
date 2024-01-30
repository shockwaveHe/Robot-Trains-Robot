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

        self.half_period = round(self.config.plan_period / self.config.control_dt / 2)

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
        self.theta = 0

    def plan_foot_steps(
        self, com_pos_target: Optional[np.ndarray] = None
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

        # Update the com trajectory based on the foot steps.
        self.com_traj, self.com_state_curr = self.pc.compute_com_traj(
            self.com_state_curr, self.foot_steps
        )
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

        self.foot_steps = self.fsp.calculate_steps(
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
            self.right_offset_delta = (
                self.right_offset_target - self.right_offset
            ) / self.half_period
            self.next_support_leg = "right"
        elif support_leg == "right":
            self.left_offset_target = offset
            self.left_offset_delta = (
                self.left_offset_target - self.left_offset
            ) / self.half_period
            self.next_support_leg = "left"

    def solve_joint_angles(self) -> Tuple[List[float], List[float], int]:
        """
        Calculate the next position of the robot based on the walking pattern.

        Returns:
            Tuple containing the next joint angles, left foot position, right foot position,
            pattern X position, and remaining pattern length.
        """
        com_attr = self.com_traj.pop(0)
        control_steps = round(
            (self.foot_steps[1].time - self.foot_steps[0].time) / self.config.control_dt
        )
        theta_change = (
            self.foot_steps[1].position[2] - self.foot_steps[0].position[2]
        ) / control_steps
        self.theta += theta_change

        if self.foot_steps[0].support_leg == "right":
            self.left_up, self.left_offset = self._get_foot_offset(
                self.left_up,
                self.left_offset,
                self.left_offset_target,
                self.left_offset_delta,
                control_steps,
            )
        elif self.foot_steps[0].support_leg == "left":
            self.right_up, self.right_offset = self._get_foot_offset(
                self.right_up,
                self.right_offset,
                self.right_offset_target,
                self.right_offset_delta,
                control_steps,
            )

        left_foot_pos, left_foot_ori = self._get_foot_position(
            self.left_sole_init, self.left_offset, self.left_up, com_attr
        )
        right_foot_pos, right_foot_ori = self._get_foot_position(
            self.right_sole_init, self.right_offset, self.right_up, com_attr
        )
        self.joint_angles = self.robot.solve_ik(
            left_foot_pos,
            left_foot_ori,
            right_foot_pos,
            right_foot_ori,
            self.joint_angles,
        )
        # projected_com_pos = [com_attr[2], com_attr[3]]
        is_control_reached = len(self.com_traj) == 0

        return self.joint_angles, is_control_reached

    def _get_foot_offset(
        self,
        foot_up: float,
        foot_offset: np.ndarray,
        foot_offset_target: np.ndarray,
        foot_offset_delta: np.ndarray,
        control_steps: int,
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
        start_up = round(
            round(self.config.plan_period / 2 / self.config.control_dt) / 2
        )
        end_up = round(control_steps / 2)
        period_up = end_up - start_up

        # Determine the period range for foot movement
        period_length = control_steps - len(self.com_traj)

        # Up or down foot movement
        if start_up < period_length <= end_up:
            foot_up += self.config.foot_step_height / period_up
        elif foot_up > 0:
            foot_up = max(foot_up - self.config.foot_step_height / period_up, 0.0)

        # Move foot in the axes of x, y, theta
        if period_length > start_up:
            foot_offset += foot_offset_delta
            if period_length > (start_up + period_up * 2):
                foot_offset = foot_offset_target.copy()

        return foot_up, foot_offset

    def _get_foot_position(
        self,
        foot_init: List[float],
        foot_offset: np.ndarray,
        foot_up: float,
        com_attr: List[float],
    ) -> List[float]:
        """
        Calculate the position of the foot based on the current offsets and height.

        Args:
            foot_init (List[float]): Initial position and orientation of the foot.
            foot_offset (np.ndarray): Offset of the foot.
            foot_up (float): Height of the foot.
            com_attr (List[float]): First pattern position.

        Returns:
            List[float]: Calculated position of the foot.
        """
        offset = foot_offset - np.block([[com_attr[0:2], 0]])
        foot_x = foot_init[0] + offset[0, 0]
        foot_y = foot_init[1] + offset[0, 1]
        foot_z = foot_init[2] + foot_up
        foot_theta = self.theta - offset[0, 2]

        return [foot_x, foot_y, foot_z], [0.0, 0.0, foot_theta]


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        choices=["sustaina_op", "robotis_op3"],
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        choices=["pybullet", "mujoco"],
        help="The simulator to use.",
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

    joint_angles, joint_names = sim.get_named_zero_joint_angles(robot)
    if robot.name == "robotis_op3":
        joint_angles[joint_names.index("l_sho_roll")] = np.pi / 4
        joint_angles[joint_names.index("r_sho_roll")] = -np.pi / 4

    walking = Walking(
        robot, config, left_foot_pos_init, right_foot_pos_init, joint_angles
    )

    sim_step_idx = 0
    foot_steps = walking.plan_foot_steps(config.target_pos_init)

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx, foot_steps, joint_angles):
        sim_step_idx += 1
        if sim_step_idx >= config.sim_step_interval:
            sim_step_idx = 0
            joint_angles, is_control_reached = walking.solve_joint_angles()
            if robot.name == "sustaina_op":
                print(f"joint_angles: {round_floats(joint_angles[7:], 6)}")
            elif robot.name == "robotis_op3":
                print(f"joint_angles: {round_floats(joint_angles, 6)}")
            else:
                raise ValueError("Unknown robot name")

            if is_control_reached:
                if len(foot_steps) <= 5:
                    target_x, target_y, theta_target = (
                        random.random() - 0.5,
                        random.random() - 0.5,
                        random.random() - 0.5,
                    )
                    print(f"Goal: ({target_x}, {target_y}, {theta_target})")
                    # target_x += 0.1
                    foot_steps = walking.plan_foot_steps(
                        np.array([target_x, target_y, theta_target])
                    )
                else:
                    foot_steps = walking.plan_foot_steps()

        sim.set_joint_angles(robot, joint_angles)
        return sim_step_idx, foot_steps, joint_angles

    sim.simulate(step_func, (sim_step_idx, foot_steps, joint_angles), args.sleep_time)


if __name__ == "__main__":
    main()
