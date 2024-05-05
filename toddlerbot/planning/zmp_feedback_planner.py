import control
import numpy as np
from scipy.interpolate import PPoly
from scipy.linalg import expm

from toddlerbot.utils.constants import GRAVITY


class ExponentialPlusPiecewisePolynomial:
    def __init__(self, K, A, alpha, ppoly):
        self.K = K
        self.A = A
        self.alpha = alpha
        self.ppoly = ppoly

    def value(self, t):
        # Evaluate the polynomial part at time t
        result = self.ppoly(t)

        # Determine the index of the segment that contains the time t
        segment_index = np.searchsorted(self.ppoly.x, t, side="right") - 1
        # Ensure the index is within the valid range
        segment_index = max(0, min(segment_index, len(self.ppoly.x) - 2))
        # Calculate the time offset from the beginning of the segment
        tj = self.ppoly.x[segment_index]
        # Compute the exponential part
        exponential = expm(self.A * (t - tj))
        # Compute result combining the polynomial and exponential parts
        result += (
            self.K @ exponential @ self.alpha[:, segment_index : segment_index + 1]
        ).flatten()

        return result

    def derivative(self, order):
        K_new = self.K
        for _ in range(order):
            K_new = K_new @ self.A

        # Derivative for exponential part needs special handling if it involves differentiation w.r.t time
        return ExponentialPlusPiecewisePolynomial(
            K_new, self.A, self.alpha, self.ppoly.derivative(order)
        )


class ZMPFeedbackPlanner:
    def __init__(self):
        self.planned = False

    def plan(self, time_steps, zmp_d, x0, com_z, Qy, R):
        self.time_steps = time_steps
        self.zmp_d = zmp_d

        # Eq. 1 and 2 in [1]
        A = np.zeros((4, 4))
        A[:2, 2:] = np.eye(2)
        B = np.zeros((4, 2))
        B[2:, :] = np.eye(2)
        self.C = np.zeros((2, 4))
        self.C[:, :2] = np.eye(2)
        self.D = -com_z / GRAVITY * np.eye(2)

        # Eq. 9 - 14 in [1]
        Q1 = self.C.T @ Qy @ self.C
        R1 = R + self.D.T @ Qy @ self.D
        N = self.C.T @ Qy @ self.D
        R1_inv = np.linalg.inv(R1)

        K, S, _ = control.lqr(A, B, Q1, R1, N)
        self.K = -K

        # Computes the time varying linear and constant term in the value function
        # and linear policy. Also known as the backward pass.
        NB = N.T + B.T @ S
        # Eq. 23, 24 in [1]
        A2 = NB.T @ R1_inv @ B.T - A.T
        B2 = 2 * (self.C.T - NB.T @ R1_inv @ self.D) @ Qy
        A2_inv = np.linalg.inv(A2)

        # Last desired ZMP
        zmp_ref_last = zmp_d[-1]
        vec4 = np.zeros(4)

        n_segments = len(zmp_d) - 1
        alpha = np.zeros((4, n_segments))
        beta = [np.zeros((4, 1)) for _ in range(n_segments)]
        gamma = [np.zeros((2, 1)) for _ in range(n_segments)]
        c = [np.zeros((2, 1)) for _ in range(n_segments)]

        # Algorithm 1 in [1] to solve for parameters of s2 and k2
        for t in range(n_segments - 1, -1, -1):
            # Assume linear interpolation between zmp points
            c[t][:, 0] = zmp_d[t] - zmp_ref_last

            # degree 4
            beta[t][:, 0] = -A2_inv @ B2 @ c[t][:, 0]
            gamma[t][:, 0] = (
                R1_inv @ self.D @ Qy @ c[t][:, 0] - 0.5 * R1_inv @ B.T @ beta[t][:, 0]
            )

            dt = time_steps[t + 1] - time_steps[t]
            A2exp = expm(A2 * dt)

            if t == n_segments - 1:
                vec4 = -beta[t]
            else:
                vec4 = alpha[:, t + 1 : t + 2] + beta[t + 1] - beta[t]

            alpha[:, t] = (np.linalg.inv(A2exp) @ vec4).squeeze()

        # (degree+1, num_vars, num_segments)
        all_beta_coeffs = np.transpose(np.stack(beta, axis=1), (2, 1, 0))
        all_gamma_coeffs = np.transpose(np.stack(gamma, axis=1), (2, 1, 0))

        # Eq. 25 in [1]
        beta_traj = PPoly(all_beta_coeffs, time_steps)
        self.s2 = ExponentialPlusPiecewisePolynomial(np.eye(4), A2, alpha, beta_traj)

        # Eq. 28 in [1]
        gamma_traj = PPoly(all_gamma_coeffs, time_steps)
        self.k2 = ExponentialPlusPiecewisePolynomial(
            -0.5 * R1_inv @ B.T, A2, alpha, gamma_traj
        )

        # for v in [0, 2, 4, 6, 8, 10]:
        #     print(f"beta: {beta_traj(v)}")
        #     print(f"gamma: {gamma_traj(v)}")
        #     print(f"s2: {self.s2.value(v)}")
        #     print(f"k2: {self.k2.value(v)}")

        # Computes the nominal CoM trajectory. Also known as the forward pass.
        # Eq. 35, 36 in [1]
        Az = np.zeros((8, 8))
        Az[:4, :4] = A + B @ self.K
        Az[:4, 4:] = -0.5 * B @ R1_inv @ B.T
        Az[4:, 4:] = A2
        Azi = np.linalg.inv(Az)
        Bz = np.zeros((8, 2))
        Bz[:4, :] = B @ R1_inv @ self.D @ Qy
        Bz[4:, :] = B2

        a = np.zeros((8, n_segments))
        a[4:, :] = alpha

        b = [np.zeros((4, 1)) for _ in range(n_segments)]
        I48 = np.zeros((4, 8))
        I48[:, :4] = np.eye(4)

        x = x0.copy()
        x[:2] -= zmp_ref_last

        # Algorithm 2 in [1] to solve for the CoM trajectory
        for t in range(n_segments):
            dt = time_steps[t + 1] - time_steps[t]
            b[t][:, 0] = -Azi[:4, :] @ Bz @ c[t][:, 0]

            a[:4, t] = x - b[t][:, 0]

            Az_exp = expm(Az * dt)
            x = I48 @ Az_exp @ a[:, t] + b[t].squeeze()

            b[t][:2, 0] += zmp_ref_last  # Map CoM position back to world frame

        mat28 = np.zeros((2, 8))
        mat28[:, :2] = np.eye(2)
        all_b_coeffs = np.transpose(np.stack(b, axis=1), (2, 1, 0))
        b_traj = PPoly(all_b_coeffs[..., :2], time_steps)

        self.com_pos = ExponentialPlusPiecewisePolynomial(mat28, Az, a, b_traj)
        self.com_vel = self.com_pos.derivative(1)
        self.com_acc = self.com_vel.derivative(1)

        # for v in [0, 2, 4, 6, 8, 10]:
        #     print(f"b: {b_traj(v)}")
        #     print(f"com_pos: {self.com_pos.value(v)}")
        #     print(f"com_vel: {self.com_vel.value(v)}")
        #     print(f"com_acc: {self.com_acc.value(v)}")

        self.planned = True

    def compute_optimal_com_acc(self, time, x):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        # Eq. 20 in [1]
        yf = self.zmp_d[-1]
        x_bar = x.copy()
        x_bar[:2] -= yf
        return self.K @ x_bar + self.k2.value(time)

    def com_acc_to_cop(self, x, u):
        if not self.planned:
            raise ValueError("Plan must be called first.")
        return self.C @ x + self.D @ u

    def get_desired_zmp_traj(self):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.time_steps, self.zmp_d

    def get_desired_zmp(self, time):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.zmp_d[np.searchsorted(self.time_steps, time, side="right") - 1]

    def get_nominal_com(self, time):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_pos.value(time)

    def get_nominal_com_vel(self, time):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_vel.value(time)

    def get_nominal_com_acc(self, time):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_acc.value(time)
