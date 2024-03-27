from dataclasses import dataclass
from typing import List, Tuple

import control
import numpy as np

from toddlerbot.planning.foot_step_planner import FootStep
from toddlerbot.utils.constants import GRAVITY


@dataclass
class ZMPPreviewControlParameters:
    """Data class to hold control parameters for preview control."""

    com_z: float  # Height of the center of mass
    dt: float  # Time step
    t_preview: float  # Control period
    Q_val: float  # Weighting for state cost
    R_val: float  # Weighting for control input cost
    x_offset_com_to_foot: float  # Offset of the center of mass from the foot
    y_disp_zmp: float  # Offset of the zero moment point from the center of mass
    filter_dynamics: bool  # Flag to filter dynamics


class ZMPPreviewController:
    def __init__(self, params: ZMPPreviewControlParameters):
        """
        Initialize the Preview Control system.
        """
        self.params = params
        self._setup()

    def _setup(self):
        """
        Set up the preview control parameters and compute the gain.
        """
        # The notations follow p.145-146 in "Introduction to Humanoid Robotics" by Shuuji Kajita

        # State-space matrices for preview control
        # dx/dt = Ax + Bu
        # y = Cx + Du
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        C = np.array([[1, 0, -self.params.com_z / GRAVITY]])
        D = np.array([[0]])

        self.n_preview = round(self.params.t_preview / self.params.dt)

        # Create a state space system.
        sys = control.ss(A, B, C, D)
        # Convert a continuous time system to discrete time by sampling
        sys_d = control.c2d(sys, self.params.dt)
        # Return state space data objects for a system
        # C doesn't change
        self.A_d, self.B_d, self.C_d, _ = control.ssdata(sys_d)

        # Define matrices for preview control
        A_tilde = np.block([[1.0, self.C_d @ self.A_d], [np.zeros((3, 1)), self.A_d]])
        B_tilde = np.block([[self.C_d @ self.B_d], [self.B_d]])
        Q = np.zeros((4, 4))
        Q[0, 0] = self.params.Q_val
        R = np.array([[self.params.R_val]])
        # Solve the discrete-time algebraic Riccati equation
        # The gain matrix is chosen to minimize the cost function
        P_tilde, _, K_tilde = control.dare(A_tilde, B_tilde, Q, R)

        self.Ks = K_tilde[0, 0]
        self.Kx = K_tilde[0, 1:]

        Ac_tilde = A_tilde - B_tilde @ K_tilde
        I_tilde = np.block([[1.0], [np.zeros((3, 1))]])
        X_tilde = -Ac_tilde.T @ P_tilde @ I_tilde

        self.G = np.zeros((1, self.n_preview))
        for i in range(self.n_preview):
            self.G[0, i] = (
                np.linalg.inv(R + B_tilde.T @ P_tilde @ B_tilde) @ (B_tilde.T) @ X_tilde
            )
            X_tilde = Ac_tilde.T @ X_tilde

    def compute_zmp_ref_traj(self, foot_steps: List[FootStep]) -> List:
        # Prepare the timing and positions matrix for all foot steps in advance
        x_offset = self.params.x_offset_com_to_foot
        y_offset = self.params.y_disp_zmp

        dt = self.params.dt
        fs_times = np.array([fs.time for fs in foot_steps])

        fs_positions = []
        for fs in foot_steps:
            x, y, theta = fs.position
            if fs.support_leg == "left":
                fs_positions.append(
                    [
                        x + np.cos(theta) * x_offset - np.sin(theta) * y_offset,
                        y + np.sin(theta) * x_offset + np.cos(theta) * y_offset,
                    ]
                )
            elif fs.support_leg == "right":
                fs_positions.append(
                    [
                        x + np.cos(theta) * x_offset + np.sin(theta) * y_offset,
                        y + np.sin(theta) * x_offset - np.cos(theta) * y_offset,
                    ]
                )
            else:
                fs_positions.append(
                    [x + np.cos(theta) * x_offset, y + np.sin(theta) * x_offset]
                )

        zmp_ref_traj = []
        i = 0
        for t in np.arange(fs_times[0], fs_times[-1] + self.params.t_preview + dt, dt):
            if t >= fs_times[-1]:
                zmp_ref_traj.append(fs_positions[-1])
            else:
                zmp_ref_traj.append(fs_positions[i])
                if (t != fs_times[0]) and (abs(t - fs_times[i + 1]) < 1e-6):
                    i += 1

        return zmp_ref_traj

    def compute_com_traj(
        self, com_curr: np.ndarray, zmp_ref: List
    ) -> Tuple[List, np.ndarray]:
        com_ref_traj = []
        zmp_simple_traj = []
        sum_error = np.zeros(2)

        for k in range(len(zmp_ref) - self.n_preview - 1):
            zmp_preview = zmp_ref[k : k + self.n_preview]
            # We calculate the ZMP assuming a cart-table model
            # Reference: Eq. 4.64 on p.138 in "Introduction to Humanoid Robotics" by Shuuji Kajita
            zmp_simple = (self.C_d @ com_curr).squeeze()
            sum_error += zmp_simple - zmp_ref[k]

            u = -self.Ks * sum_error - self.Kx @ com_curr - self.G @ zmp_preview
            com_curr = self.A_d @ com_curr + self.B_d @ u

            # Record the current COM position
            zmp_simple_traj.append(zmp_simple)
            com_ref_traj.append(com_curr[0].copy())

        return zmp_simple_traj, com_ref_traj, com_curr
