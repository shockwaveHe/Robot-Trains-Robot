from dataclasses import dataclass
from typing import List, Tuple

import control
import numpy as np

from toddleroid.planning.foot_step_planner import FootStep, Position
from toddleroid.utils.constants import GRAVITY
from toddleroid.utils.data_utils import round_floats


@dataclass
class ControlParameters:
    """Data class to hold control parameters for preview control."""

    dt: float  # Time step
    period: float  # Control period
    Q_val: float  # Weighting for state cost
    H_val: float  # Weighting for control input cost


@dataclass
class StateSpace:
    """Data class to hold state space representation matrices."""

    A: np.ndarray  # State transition matrix
    B: np.ndarray  # Control input matrix
    C: np.ndarray  # Output matrix
    D: np.ndarray  # Direct transmission matrix


class PreviewControl:
    def __init__(self, params: ControlParameters, state_space: StateSpace):
        """
        Initialize the Preview Control system.

        Args:
            params (ControlParameters): Control parameters including dt, period, Q_val, and H_val.
            state_space (StateSpace): State space matrices A, B, C, D.
        """
        self.params = params
        self.state_space = state_space
        self._setup()

    def _setup(self):
        """
        Set up the preview control parameters and compute the LQR gain.
        """
        # Discretize the continuous state-space system
        sys_d = control.c2d(
            control.ss(
                self.state_space.A,
                self.state_space.B,
                self.state_space.C,
                self.state_space.D,
            ),
            self.params.dt,
        )
        self.A_d, self.B_d, self.C_d, _ = control.ssdata(sys_d)

        # Define the weighting matrices for the LQR problem
        Q_m = np.zeros((4, 4))
        Q_m[0, 0] = self.params.Q_val
        H = np.array([[self.params.H_val]])

        # Define extended system matrices for preview control
        Phai = np.block([[1.0, -self.C_d @ self.A_d], [np.zeros((3, 1)), self.A_d]])
        G = np.block([[-self.C_d @ self.B_d], [self.B_d]])
        GR = np.block([[1.0], [np.zeros((3, 1))]])

        # Solve the discrete-time algebraic Riccati equation
        P = control.dare(Phai, G, Q_m, H)[0]
        # Compute the LQR gain
        # TODO: Understand this part
        self.F = -np.linalg.inv(H + G.transpose() @ P @ G) @ G.transpose() @ P @ Phai
        xi = Phai + G @ self.F
        self.f = [
            -np.linalg.inv(H + G.transpose() @ P @ G)
            @ G.transpose()
            @ np.linalg.matrix_power(xi.transpose(), i - 1)
            @ P
            @ GR
            for i in range(round(self.params.period / self.params.dt))
        ]

        # Initialize state variables
        self.xp, self.yp = np.zeros((3, 1)), np.zeros((3, 1))
        self.ux, self.uy = 0.0, 0.0

    def compute_control_pattern(
        self,
        t: float,
        current_x: np.ndarray,
        current_y: np.ndarray,
        foot_steps: List[FootStep],
        pre_reset: bool = False,
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Update the parameters based on the current position and foot steps.
        Args:
            t (float): Current time.
            current_x (np.ndarray): Current x position.
            current_y (np.ndarray): Current y position.
            foot_steps (List[FootStep]): List of footsteps.
            pre_reset (bool, optional): Flag to reset previous x and y. Defaults to False.

        Returns:
            Tuple[List, np.ndarray, np.ndarray]:
                - List of Center of Mass positions.
                - Updated x position.
                - Updated y position.
        """
        # Initialize or reset current position and control inputs
        x, y = current_x.copy(), current_y.copy()
        if pre_reset:
            self.xp, self.yp = x.copy(), y.copy()
            self.ux, self.uy = 0.0, 0.0

        COM_list = []
        num_steps = round((foot_steps[1].time - t) / self.params.dt)

        # Calculate the center of mass positions
        for i in range(num_steps):
            # Compute error and state difference
            px, py = np.dot(self.C_d, x), np.dot(self.C_d, y)
            ex, ey = foot_steps[0].position.x - px, foot_steps[0].position.y - py
            X, Y = np.vstack([ex, x - self.xp]), np.vstack([ey, y - self.yp])
            self.xp, self.yp = x.copy(), y.copy()

            # Update control inputs based on LQR gain
            dux, duy = np.dot(self.F, X), np.dot(self.F, Y)

            # Iterate through footstep timing
            index = 1
            for j in range(1, round(self.params.period / self.params.dt) - 1):
                step_time = round((i + j) + t / self.params.dt)
                if step_time >= round(foot_steps[index].time / self.params.dt):
                    dux += self.f[j] * (
                        foot_steps[index].position.x - foot_steps[index - 1].position.x
                    )
                    duy += self.f[j] * (
                        foot_steps[index].position.y - foot_steps[index - 1].position.y
                    )
                    index += 1

            # Update the state based on system dynamics
            self.ux += dux
            self.uy += duy
            x = np.dot(self.A_d, x) + np.dot(self.B_d, self.ux)
            y = np.dot(self.A_d, y) + np.dot(self.B_d, self.uy)

            # Append the current COM positions to the list
            COM_list.append(np.hstack([x[0, 0], y[0, 0], px[0, 0], py[0, 0]]))

        return COM_list, x, y


# Example usage
if __name__ == "__main__":
    foot_steps = [
        FootStep(time=0, position=Position(x=0, y=0)),
        FootStep(time=0.34, position=Position(x=0, y=0.06)),
        FootStep(time=0.68, position=Position(x=0.05, y=-0.04)),
        FootStep(time=1.02, position=Position(x=0.10, y=0.1)),
        FootStep(time=1.36, position=Position(x=0.15, y=0.0)),
        FootStep(time=1.7, position=Position(x=0.20, y=0.14)),
        FootStep(time=2.04, position=Position(x=0.25, y=0.1)),
        FootStep(time=2.72, position=Position(x=0.25, y=0.1)),
        FootStep(time=100, position=Position(x=0.25, y=0.1)),
    ]
    x, y = np.zeros((3, 1)), np.zeros((3, 1))

    control_params = ControlParameters(dt=0.01, period=1.0, Q_val=1e8, H_val=1.0)

    robot_height = 0.27
    # State-space matrices for preview control
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0], [0], [1]])
    C = np.array([[1, 0, -robot_height / GRAVITY]])
    D = np.array([[0]])
    state_space = StateSpace(A, B, C, D)

    pc = PreviewControl(control_params, state_space)

    for i in range(len(foot_steps) - 2):
        com, x, y = pc.compute_control_pattern(foot_steps[i].time, x, y, foot_steps[i:])
        print(round_floats(com[-1], 6))

    print("Preview control simulation completed.")
