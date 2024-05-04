import control
import numpy as np

from toddlerbot.utils.misc_utils import profile


class LQRFullBodyController:
    def __init__(self, nq, dt, Q_val, R_val):
        # Placeholder for future initialization
        self.nq = nq
        self.dt = dt
        self.Q = Q_val * np.eye(2 * nq)
        self.R = R_val * np.eye(nq)

    @profile()
    def _compute_lqr_gain(self, mass_matrix, bias_forces):
        # Define system matrices
        A = np.zeros((2 * self.nq, 2 * self.nq))
        B = np.zeros((2 * self.nq, self.nq))
        A[: self.nq, self.nq :] = np.eye(self.nq)
        C = np.eye(2 * self.nq)  # Output matrix maps state directly
        D = np.zeros((2 * self.nq, self.nq))  # No direct feedthrough

        try:
            dynamic_part = -np.linalg.solve(mass_matrix, bias_forces)
            A[self.nq :, : self.nq] = dynamic_part.reshape(self.nq, -1)
            B[self.nq :, :] = np.linalg.inv(mass_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix is singular or ill-conditioned")

        # Create the state-space model and calculate LQR gain
        dsys = control.ss(A, B, C, D, self.dt)
        K, _, _ = control.dlqr(dsys, self.Q, self.R)
        return K

    @profile()
    def control(self, qpos, qvel, qpos_ref, qvel_ref, mass_matrix, bias_forces):
        if qpos.shape[0] != qvel.shape[0]:
            raise ValueError("Position and velocity vectors must be of the same length")

        x = np.concatenate([qpos - qpos_ref, qvel - qvel_ref])
        K = self._compute_lqr_gain(mass_matrix, bias_forces)
        u = -K @ x  # Control input considering deviation from reference
        return u
