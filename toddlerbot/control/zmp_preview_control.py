from dataclasses import dataclass
from typing import List, Tuple

import control
import numpy as np

from toddlerbot.planning.foot_step_planner import FootStep
from toddlerbot.utils.constants import GRAVITY


@dataclass
class ZMPPreviewControlParameters:
    """Data class to hold control parameters for preview control."""

    com_height: float  # Height of the center of mass
    dt: float  # Time step
    period: float  # Control period
    Q_val: float  # Weighting for state cost
    R_val: float  # Weighting for control input cost


class ZMPPreviewController:
    def __init__(self, params: ZMPPreviewControlParameters):
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
        Set up the preview control parameters and compute the gain.
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
        dare_sol_mat, _, gain_mat = control.dare(extended_A, extended_B, Q, R)
        self.gain = -gain_mat

        # Compute the gain for the closed-loop system
        closed_loop_system_mat = extended_A + extended_B @ self.gain
        # Define the reference input matrix for preview control
        reference_input_mat = np.block([[1.0], [np.zeros((3, 1))]])
        # Calculate the preview control gains for a defined period based on sampling time
        self.preview_control_gains = []
        for i in range(round(self.params.period / self.params.dt)):
            preview_control_gain = (
                -np.linalg.inv(R + extended_B.transpose() @ dare_sol_mat @ extended_B)
                @ extended_B.transpose()
                @ np.linalg.matrix_power(closed_loop_system_mat.transpose(), i)
                @ dare_sol_mat
                @ reference_input_mat
            ).item()
            self.preview_control_gains.append(preview_control_gain)

        self.preview_control_gains = np.array(self.preview_control_gains)

        # Initialize state variables on x and y axis
        self.com_state_prev = np.zeros((3, 2))
        self.control_input = np.zeros(2)

    def compute_com_traj(
        self,
        com_state_curr: np.ndarray,
        foot_steps: List[FootStep],
        reset: bool = False,
    ) -> Tuple[List, np.ndarray]:
        """Calculate the trajectory of the center of mass (COM) based on the current state and foot steps."""
        if reset:
            self.com_state_prev = com_state_curr.copy()
            self.control_input = np.zeros(2)

        # Prepare the timing and positions matrix for all foot steps in advance
        times = np.array([fs.time for fs in foot_steps])
        positions = np.stack([fs.position[:2] for fs in foot_steps])

        # Calculate the number of steps for the control loop based on the first two foot steps
        control_steps = round((times[1] - times[0]) / self.params.dt)
        preview_steps = round(self.params.period / self.params.dt)

        com_traj = []
        for step_index in range(control_steps):
            # Calculate the current footstep's position error
            state_error = positions[0] - self.C_d @ com_state_curr
            state_diff = np.concatenate(
                [state_error, com_state_curr - self.com_state_prev]
            )
            self.com_state_prev = com_state_curr.copy()

            control_update = (self.gain @ state_diff).squeeze()
            index = 1
            # Adjust control based on footstep timing
            for j in range(preview_steps - 2):
                future_steps = round((times[index] - times[0]) / self.params.dt)
                if step_index + j >= future_steps - 1:
                    control_update += self.preview_control_gains[j] * (
                        positions[index] - positions[index - 1]
                    )
                    index += 1
                    print(f"step_index: {step_index}, j: {j}, index: {index}")

            self.control_input += control_update
            com_state_curr = (
                self.A_d @ com_state_curr + self.B_d @ self.control_input[None]
            )

            # Record the current COM position
            com_traj.append(com_state_curr[0].copy())

        return com_traj, com_state_curr
