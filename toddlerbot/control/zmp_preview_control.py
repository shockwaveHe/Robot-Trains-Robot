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
    y_offset_com_to_foot: float  # Offset of the center of mass from the foot
    y_offset_zmp: float  # Offset of the zero moment point from the center of mass


class ZMPPreviewController:
    def __init__(self, params: ZMPPreviewControlParameters):
        """
        Initialize the Preview Control system.

        Args:
            params (ControlParameters): Control parameters including dt, period, Q_val, and R_val.
            com_z (float): height of the center of mass.
        """
        self.params = params
        self._setup()

    def _setup(self):
        """
        Set up the preview control parameters and compute the gain.
        """

        # The notations follow p.145-146 of "Introduction to Humanoid Robotics" by Shuuji Kajita

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

    def compute_com_traj(
        self, com_curr: np.ndarray, foot_steps: List[FootStep]
    ) -> Tuple[List, np.ndarray]:
        # Prepare the timing and positions matrix for all foot steps in advance
        dt = self.params.dt
        fs_times = np.array([fs.time for fs in foot_steps])

        y_offset_disp = self.params.y_offset_zmp - self.params.y_offset_com_to_foot
        fs_positions = []
        for fs in foot_steps:
            x, y, theta = fs.position
            if fs.support_leg == "left":
                fs_positions.append(
                    [
                        x - np.sin(theta) * y_offset_disp,
                        y + np.cos(theta) * y_offset_disp,
                    ]
                )
            elif fs.support_leg == "right":
                fs_positions.append(
                    [
                        x + np.sin(theta) * y_offset_disp,
                        y - np.cos(theta) * y_offset_disp,
                    ]
                )
            else:
                fs_positions.append([x, y])

        zmp_ref = []
        i = 0
        for t in np.arange(fs_times[0], fs_times[-1] + self.params.t_preview + dt, dt):
            if t >= fs_times[-1]:
                zmp_ref.append(fs_positions[-1])
            else:
                zmp_ref.append(fs_positions[i])
                if (t != fs_times[0]) and (abs(t - fs_times[i + 1]) < 1e-6):
                    i += 1

        n_sim = round((fs_times[-1] - fs_times[0]) / dt)
        sum_error = np.zeros(2)
        com_traj = []
        zmp_traj = []
        for k in range(n_sim):
            zmp_preview = zmp_ref[k : k + self.n_preview]

            zmp = (self.C_d @ com_curr).squeeze()
            sum_error += zmp - zmp_ref[k]

            u = -self.Ks * sum_error - self.Kx @ com_curr - self.G @ zmp_preview
            com_curr = self.A_d @ com_curr + self.B_d @ u

            # Record the current COM position
            zmp_traj.append(zmp)
            com_traj.append(com_curr[0].copy())

        return zmp_ref, zmp_traj, com_traj, com_curr
