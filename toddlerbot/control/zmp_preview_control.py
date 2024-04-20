from dataclasses import dataclass
from typing import List

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
    t_filter: float  # Filter time constant
    Q_val: float  # Weighting for state cost
    R_val: float  # Weighting for control input cost
    x_offset_com_to_foot: float  # Offset of the center of mass from the foot
    y_disp_zmp: float  # Offset of the zero moment point from the center of mass


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

        self.n_preview = round(self.params.t_preview / self.params.dt)
        # self.n_filter = round(self.params.t_filter / self.params.dt)

        # State-space matrices
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        C = np.array([[1, 0, -self.params.com_z / GRAVITY]])
        D = np.array([[0]])
        Q = np.array([[self.params.Q_val]])
        R = np.array([[self.params.R_val]])

        # Convert a continuous time system to discrete time by sampling
        # Return state space data objects for a system
        sys = control.ss(A, B, C, D)
        sys_d = control.c2d(sys, self.params.dt)
        A_d, B_d, C_d, _ = control.ssdata(sys_d)  # C doesn't change
        self.A_d, self.B_d, self.C_d = A_d, B_d, C_d

        P, _, self.K = control.dare(A_d, B_d, C_d.T @ Q @ C_d, R)

        self.f = np.zeros((1, self.n_preview))
        for i in range(self.n_preview):
            self.f[0, i] = (
                np.linalg.inv(R + B_d.T @ P @ B_d)
                @ B_d.T
                @ np.linalg.matrix_power((A_d - B_d @ self.K).T, i)
                @ C_d.T
                @ Q
            )

        # Define matrices for the improved preview control
        A_tilde = np.block([[1.0, C_d @ A_d], [np.zeros((3, 1)), A_d]])
        B_tilde = np.block([[C_d @ B_d], [B_d]])
        Q_tilde = np.zeros((4, 4))
        Q_tilde[0, 0] = self.params.Q_val
        # Solve the discrete-time algebraic Riccati equation
        # The gain matrix is chosen to minimize the cost function
        P_tilde, _, K_tilde = control.dare(A_tilde, B_tilde, Q_tilde, R)

        self.Ks = K_tilde[0, 0]
        self.Kx = K_tilde[0, 1:]

        Ac_tilde = A_tilde - B_tilde @ K_tilde
        I_tilde = np.block([[1.0], [np.zeros((3, 1))]])
        X_tilde = -Ac_tilde.T @ P_tilde @ I_tilde

        self.G = np.zeros((1, self.n_preview))
        for i in range(self.n_preview):
            self.G[0, i] = (
                np.linalg.inv(R + B_tilde.T @ P_tilde @ B_tilde) @ B_tilde.T @ X_tilde
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

    def compute_com_traj(self, com_curr, zmp_ref_traj):
        zmp_traj = []
        com_ref_traj = []
        sum_error = np.zeros(2)
        for k in range(len(zmp_ref_traj) - self.n_preview - 1):
            zmp_preview = zmp_ref_traj[k : k + self.n_preview]
            # We calculate the ZMP assuming a cart-table model
            # Reference: Eq. 4.64 on p.138 in "Introduction to Humanoid Robotics" by Shuuji Kajita
            zmp = (self.C_d @ com_curr).squeeze()
            sum_error += zmp - zmp_ref_traj[k]

            u = -self.Ks * sum_error - self.Kx @ com_curr - self.G @ zmp_preview
            com_curr = self.A_d @ com_curr + self.B_d @ u

            # Record the current COM position
            zmp_traj.append(zmp)
            com_ref_traj.append(com_curr[0].copy())

        return zmp_traj, com_ref_traj

    # TODO: Implement the foot step modification function
