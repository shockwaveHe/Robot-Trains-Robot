from dataclasses import dataclass
from typing import List

import numpy as np

from toddleroid.utils.data_utils import round_floats


@dataclass
class FootStepPlanParameters:
    max_stride: np.ndarray  # x, y, theta
    period: float
    width: float


@dataclass
class FootStep:
    time: float
    position: np.ndarray  # x, y, theta
    support_leg: str = ""


class FootStepPlanner:
    def __init__(self, params: FootStepPlanParameters):
        """
        Initialize the foot step planner with given parameters.

        Args:
            params (PlanParameters): Parameters for foot step planning.
        """
        self.params = params

    def calculate_steps(
        self,
        target: np.ndarray,
        current: np.ndarray,
        next_support_leg: str,
        status: str,
    ) -> List[FootStep]:
        """
        Calculate a series of foot steps to reach the target position.

        Args:
            target (np.ndarray): Goal position and orientation.
            current (np.ndarray): Current position and orientation.
            next_support_leg (str): The next leg to move ('left' or 'right').
            status (str): The status of the robot ('start', 'walking', 'stop').

        Returns:
            List[Step]: A list of steps to reach the target position.
        """
        steps = []
        time = 0.0
        stride = self._calculate_strides(target, current)

        if status == "start":
            steps.append(FootStep(time, current, "both"))
            time += self.params.period * 2.0

        if next_support_leg in ["left", "right"]:
            steps.append(self._create_step(time, current, next_support_leg))
            next_support_leg = "left" if next_support_leg == "right" else "right"

        while not self._is_target_reached(target, current):
            time += self.params.period
            current = current + stride
            steps.append(self._create_step(time, current, next_support_leg))
            next_support_leg = "left" if next_support_leg == "right" else "right"

        self._add_final_steps(steps, target, time, next_support_leg, status)
        return steps

    def _calculate_strides(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Calculate the stride values in x, y, and theta directions to move from the current position to the target.

        Args:
            target (np.ndarray): The target position and orientation, represented as x, y, and theta.
            current (np.ndarray): The current position and orientation.

        Returns:
            np.ndarray: The stride in x, y, and theta necessary to move towards the target while respecting
                    the maximum allowed strides.
        """
        max_step = max(np.abs(target - current) / self.params.max_stride)
        return (target - current) / max_step

    def _is_target_reached(self, target: np.ndarray, current: np.ndarray) -> bool:
        """
        Check if the target position is reached.

        Args:
            target (np.ndarray): Goal position and orientation.
            current (np.ndarray): Current position and orientation.

        Returns:
            bool: True if the target is reached, False otherwise.
        """
        return np.all(np.abs(target - current) <= self.params.max_stride)

    def _create_step(
        self, time: float, current: np.ndarray, support_leg: str
    ) -> FootStep:
        """
        Create a single foot step with adjusted y position based on support leg.

        Args:
            time (float): Time at which the step occurs.
            current (np.ndarray): Current position.
            support_leg (str): The supporting leg ('left' or 'right').

        Returns:
            Step: The foot step as a dataclass instance.
        """
        adjusted_y = (
            current[1] + self.params.width
            if support_leg == "left"
            else current[1] - self.params.width
        )
        return FootStep(
            time, np.array([current[0], adjusted_y, current[2]]), support_leg
        )

    def _add_final_steps(
        self,
        steps: List[FootStep],
        target: np.ndarray,
        time: float,
        next_support_leg: str,
        status: str,
    ) -> None:
        """
        Add final steps to reach the target position.

        Args:
            steps (List[Step]): List of steps generated so far.
            target (np.ndarray): Goal position.
            time (float): Current time in the step sequence.
            next_support_leg (str): The next leg to move ('left' or 'right').
            status (str): The status of the robot ('start', 'walking', 'stop').
        """
        if not status == "stop":
            time += self.params.period
            adjusted_target_y = (
                target[1] + self.params.width
                if next_support_leg == "left"
                else target[1] - self.params.width
            )
            steps.append(
                FootStep(
                    time,
                    np.array([target[0], adjusted_target_y, target[2]]),
                    next_support_leg,
                )
            )
            next_support_leg = "both"
            time += self.params.period

        steps.append(FootStep(time, target, next_support_leg))
        time += 100.0  # Arbitrary time to indicate the robot is stationary
        steps.append(FootStep(time, target, next_support_leg))


# Example usage
if __name__ == "__main__":
    planner_params = FootStepPlanParameters(
        max_stride=np.array([0.06, 0.04, 0.1]),
        period=0.34,
        width=0.044,
    )
    planner = FootStepPlanner(planner_params)
    foot_steps = planner.calculate_steps(
        target=np.array([1.0, 0.0, 0.5]),
        current=np.array([0.5, 0.0, 0.1]),
        next_support_leg="right",
        status="start",
    )
    for step in foot_steps:
        print(round_floats(step, 4))
