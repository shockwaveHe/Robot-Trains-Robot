from dataclasses import dataclass
from typing import List

from toddleroid.utils.data_utils import round_floats


@dataclass
class PlanParameters:
    max_stride_x: float
    max_stride_y: float
    max_stride_th: float
    period: float
    width: float


@dataclass
class Position:
    x: float
    y: float
    theta: float = 0.0


@dataclass
class FootStep:
    time: float
    position: Position
    support_leg: str = ""


class FootStepPlanner:
    def __init__(self, params: PlanParameters):
        """
        Initialize the foot step planner with given parameters.

        Args:
            params (PlanParameters): Parameters for foot step planning.
        """
        self.params = params

    def calculate_steps(
        self,
        goal: Position,
        current: Position,
        next_support_leg: str,
        status: str,
    ) -> List[FootStep]:
        """
        Calculate a series of foot steps to reach the goal position.

        Args:
            goal (FootPosition): Goal position and orientation.
            current (FootPosition): Current position and orientation.
            next_support_leg (str): The next leg to move ('left' or 'right').
            status (str): The status of the robot ('start', 'walking', 'stop').

        Returns:
            List[Step]: A list of steps to reach the goal position.
        """
        steps = []
        time = 0.0
        stride = self._calculate_strides(goal, current)

        if status == "start":
            steps.append(FootStep(time, current, "both"))
            time += self.params.period * 2.0

        if next_support_leg in ["left", "right"]:
            steps.append(self._create_step(time, current, next_support_leg))
            next_support_leg = "left" if next_support_leg == "right" else "right"

        while not self._is_goal_reached(goal, current):
            time += self.params.period
            current = Position(
                current.x + stride.x, current.y + stride.y, current.theta + stride.theta
            )
            steps.append(self._create_step(time, current, next_support_leg))
            next_support_leg = "left" if next_support_leg == "right" else "right"

        self._add_final_steps(steps, goal, time, next_support_leg, status)
        return steps

    def _calculate_strides(self, goal: Position, current: Position) -> Position:
        """
        Calculate the stride values in x, y, and theta directions to move from the current position to the goal.

        Args:
            goal (Position): The goal position and orientation, represented as a Position object with x, y, and theta.
            current (Position): The current position and orientation, also represented as a Position object.

        Returns:
            Position: A Position object representing the stride in x, y, and theta necessary to move towards the goal
                    while respecting the maximum allowed strides.
        """
        max_step = max(
            abs(goal.x - current.x) / self.params.max_stride_x,
            abs(goal.y - current.y) / self.params.max_stride_y,
            abs(goal.theta - current.theta) / self.params.max_stride_th,
        )
        return Position(
            (goal.x - current.x) / max_step,
            (goal.y - current.y) / max_step,
            (goal.theta - current.theta) / max_step,
        )

    def _is_goal_reached(self, goal: Position, current: Position) -> bool:
        """
        Check if the goal position is reached.

        Args:
            goal (FootPosition): Goal position and orientation.
            current (FootPosition): Current position and orientation.

        Returns:
            bool: True if the goal is reached, False otherwise.
        """
        return (
            abs(goal.x - current.x) <= self.params.max_stride_x
            and abs(goal.y - current.y) <= self.params.max_stride_y
            and abs(goal.theta - current.theta) <= self.params.max_stride_th
        )

    def _create_step(
        self, time: float, current: Position, support_leg: str
    ) -> FootStep:
        """
        Create a single foot step with adjusted y position based on support leg.

        Args:
            time (float): Time at which the step occurs.
            current (FootPosition): Current position.
            support_leg (str): The supporting leg ('left' or 'right').

        Returns:
            Step: The foot step as a dataclass instance.
        """
        adjusted_y = (
            current.y + self.params.width
            if support_leg == "left"
            else current.y - self.params.width
        )
        return FootStep(
            time, Position(current.x, adjusted_y, current.theta), support_leg
        )

    def _add_final_steps(
        self,
        steps: List[FootStep],
        goal: Position,
        time: float,
        next_support_leg: str,
        status: str,
    ) -> None:
        """
        Add final steps to reach the goal position.

        Args:
            steps (List[Step]): List of steps generated so far.
            goal (FootPosition): Goal position.
            time (float): Current time in the step sequence.
            next_support_leg (str): The next leg to move ('left' or 'right').
            status (str): The status of the robot ('start', 'walking', 'stop').
        """
        if not status == "stop":
            time += self.params.period
            adjusted_goal_y = (
                goal.y + self.params.width
                if next_support_leg == "left"
                else goal.y - self.params.width
            )
            steps.append(
                FootStep(
                    time,
                    Position(goal.x, adjusted_goal_y, goal.theta),
                    next_support_leg,
                )
            )
            next_support_leg = "both"
            time += self.params.period

        steps.append(FootStep(time, goal, next_support_leg))
        time += 100.0  # Arbitrary time to indicate the robot is stationary
        steps.append(FootStep(time, goal, next_support_leg))


if __name__ == "__main__":
    planner_params = PlanParameters(
        max_stride_x=0.06,
        max_stride_y=0.04,
        max_stride_th=0.1,
        period=0.34,
        width=0.044,
    )
    planner = FootStepPlanner(planner_params)
    foot_steps = planner.calculate_steps(
        goal=Position(1.0, 0.0, 0.5),
        current=Position(0.5, 0.0, 0.1),
        next_support_leg="right",
        status="start",
    )
    for step in foot_steps:
        print(round_floats(step, 4))
