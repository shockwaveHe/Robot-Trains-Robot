from dataclasses import dataclass
from typing import List, Tuple

import control
import numpy as np

from toddleroid.control.base_controller import BaseController
from toddleroid.planning.foot_step_planner import FootStep
from toddleroid.utils.constants import GRAVITY
from toddleroid.utils.data_utils import round_floats


@dataclass
class LQRPreviewControlParameters:
    """Data class to hold control parameters for preview control."""

    com_height: float  # Height of the center of mass
    dt: float  # Time step
    period: float  # Control period
    Q_val: float  # Weighting for state cost
    R_val: float  # Weighting for control input cost


class LQRPreviewController(BaseController):
    def __init__(self, params: LQRPreviewControlParameters):
        """
        Initialize the Preview Control system.

        Args:
            params (ControlParameters): Control parameters including dt, period, Q_val, and R_val.
            com_height (float): height of the center of mass.
        """
        self.params = params
        self._setup()

    def _setup(self):
        """
        Set up the preview control parameters and compute the LQR gain.
        """
        # Discretize the continuous state-space system

        # State-space matrices for preview control
        # dx/dt = Ax + Bu
        # y = Cx + Du
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        C = np.array([[1, 0, -self.params.com_height / GRAVITY]])
        D = np.array([[0]])

        # Create a state space system.
        sys = control.ss(A, B, C, D)
        # Convert a continuous time system to discrete time by sampling
        sys_d = control.c2d(sys, self.params.dt)
        # Return state space data objects for a system
        # C doesn't change
        self.A_d, self.B_d, self.C_d, _ = control.ssdata(sys_d)

        # Define extended system matrices for preview control
        extended_A = np.block(
            [[1.0, -self.C_d @ self.A_d], [np.zeros((3, 1)), self.A_d]]
        )
        extended_B = np.block([[-self.C_d @ self.B_d], [self.B_d]])
        # Define the weighting matrices for the LQR problem
        Q = np.zeros((4, 4))
        Q[0, 0] = self.params.Q_val
        R = np.array([[self.params.R_val]])

        # Solve the discrete-time algebraic Riccati equation
        # The gain matrix is chosen to minimize the cost function
        dare_sol_mat, _, self.lgr_gain = control.dare(extended_A, extended_B, Q, R)

        # Compute the LQR gain for the closed-loop system
        closed_loop_system_mat = extended_A - extended_B @ self.lgr_gain
        # Define the reference input matrix for preview control
        reference_input_mat = np.block([[1.0], [np.zeros((3, 1))]])
        # Calculate the preview control gains for a defined period based on sampling time
        self.preview_control_gains = []
        for i in range(round(self.params.period / self.params.dt)):
            preview_control_gain = (
                -np.linalg.inv(R + extended_B.transpose() @ dare_sol_mat @ extended_B)
                @ extended_B.transpose()
                @ np.linalg.matrix_power(closed_loop_system_mat.transpose(), i - 1)
                @ dare_sol_mat
                @ reference_input_mat
            ).item()
            self.preview_control_gains.append(preview_control_gain)

        # Initialize state variables on x and y axis
        self.com_state_prev = np.zeros((3, 2))
        self.u = np.zeros(2)

    def compute_control_pattern(
        self,
        t: float,
        com_state_curr: np.ndarray,
        foot_steps: List[FootStep],
        reset: bool = False,
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Update the parameters based on the current position and foot steps.
        Args:
            t (float): Current time.
            com_state_curr (np.ndarray): Current x and y position.
            foot_steps (List[FootStep]): List of footsteps.
            reset (bool, optional): Flag to reset previous x and y. Defaults to False.

        Returns:
            Tuple[List, np.ndarray, np.ndarray]:
                - List of Center of Mass positions.
                - Updated x position.
                - Updated y position.
        """
        # Initialize or reset current position and control inputs
        com_state = com_state_curr.copy()

        if reset:
            self.com_state_prev = com_state.copy()
            self.u = np.zeros(2)

        com_list = []
        # Calculate the center of mass positions
        for i in range(round((foot_steps[1].time - t) / self.params.dt)):
            # Compute error and state difference
            p = self.C_d @ com_state
            e = foot_steps[0].position[:2] - p
            X = np.concatenate([e, com_state - self.com_state_prev])
            self.com_state_prev = com_state.copy()

            # Update control inputs based on LQR gain
            du = (-self.lgr_gain @ X).squeeze()

            # Iterate through footstep timing
            index = 1
            for j in range(1, round(self.params.period / self.params.dt) - 1):
                step_time = round((i + j) + t / self.params.dt)
                if step_time >= round(foot_steps[index].time / self.params.dt):
                    du += self.preview_control_gains[j] * (
                        foot_steps[index].position[:2]
                        - foot_steps[index - 1].position[:2]
                    )
                    index += 1

            # Update the state based on system dynamics
            self.u += du
            com_state = self.A_d @ com_state + self.B_d @ self.u[None]

            # Append the current com positions to the list
            com_list.append(np.hstack([com_state[:1], p]).squeeze())

        return com_list, com_state


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

    control_params = LQRPreviewControlParameters(
        com_height=0.27, dt=0.01, period=1.0, Q_val=1e8, R_val=1.0
    )

    pc = LQRPreviewController(control_params)

    for i in range(len(foot_steps) - 2):
        com, x, y = pc.compute_control_pattern(foot_steps[i].time, x, y, foot_steps[i:])
        print(round_floats(com[-1], 6))

    print("Preview control simulation completed.")
