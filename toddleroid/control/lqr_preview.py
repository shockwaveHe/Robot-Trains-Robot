from dataclasses import dataclass
from typing import List, Tuple

import control
import numpy as np

from toddleroid.control.base_controller import BaseController
from toddleroid.planning.foot_step_planner import FootStep
from toddleroid.utils.constants import GRAVITY


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
        self.control_input = np.zeros(2)

    def compute_com_traj(
        self,
        com_state_curr: np.ndarray,
        foot_steps: List[FootStep],
        reset: bool = False,
    ) -> Tuple[List, np.ndarray]:
        """
        Calculate the trajectory of the center of mass (COM) based on the current state and foot steps.

        Args:
            com_state_curr (np.ndarray): Current state of the COM.
            foot_steps (List[FootStep]): List of planned foot steps.
            reset (bool, optional): Flag to reset previous COM state. Defaults to False.

        Returns:
            Tuple[List, np.ndarray]:
                - List of predicted COM positions.
                - Updated COM state.
        """
        com_state = com_state_curr.copy()

        if reset:
            self.com_state_prev = com_state.copy()
            self.control_input = np.zeros(2)

        com_traj = []
        for step_index in range(
            round((foot_steps[1].time - foot_steps[0].time) / self.params.dt)
        ):
            # Compute the projection of COM state
            projected_com = self.C_d @ com_state
            error = foot_steps[0].position[:2] - projected_com
            state_diff = np.concatenate([error, com_state - self.com_state_prev])
            self.com_state_prev = com_state.copy()

            # Update control inputs based on LQR gain
            control_update = (-self.lgr_gain @ state_diff).squeeze()

            index = 1
            # Adjust control based on footstep timing
            for j in range(1, round(self.params.period / self.params.dt) - 1):
                step_index_future = round(
                    (step_index + j) + foot_steps[0].time / self.params.dt
                )
                # This condition checks if the future step index has reached or surpassed
                # the time of the next footstep. The time of the footstep is converted to an index
                # by dividing by the time step and rounding.
                if step_index_future >= round(foot_steps[index].time / self.params.dt):
                    control_update += self.preview_control_gains[j] * (
                        foot_steps[index].position[:2]
                        - foot_steps[index - 1].position[:2]
                    )
                    # Increment the index to consider the next footstep in subsequent iterations.
                    index += 1

            # Apply control update to the state
            self.control_input += control_update
            com_state = self.A_d @ com_state + self.B_d @ self.control_input[None]

            # Record the current COM position
            com_traj.append(np.hstack([com_state[:1], projected_com]).squeeze())

        return com_traj, com_state
