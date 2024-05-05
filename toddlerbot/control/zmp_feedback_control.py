import control
import numpy as np
from scipy.interpolate import PPoly
from scipy.linalg import expm


class ExponentialPlusPiecewisePolynomial:
    def __init__(self, K, A, alpha, polynomial_parts):
        # A: Matrix used in the exponential calculation
        # alpha: Array of coefficients that interact with the exponential term, per segment
        # polynomial_parts: List of PPoly objects representing polynomial behavior per segment
        self.K = K
        self.A = A
        self.alpha = alpha
        self.polynomial_parts = polynomial_parts

    def value(self, t):
        # Determine which segment we are in based on time t
        for i, poly in enumerate(self.polynomial_parts):
            if poly.x[0] <= t <= poly.x[-1]:
                segment_index = i
                break
        # Evaluate the polynomial part
        poly_value = self.polynomial_parts[segment_index](t)
        # Calculate the time offset from the beginning of the segment
        tj = self.polynomial_parts[segment_index].x[0]
        # Compute the exponential part
        exponential = expm(self.A * (t - tj))
        # Combine the results
        result = poly_value + self.K @ exponential @ self.alpha[
            :, segment_index
        ].reshape(-1, 1)

        return result

    def derivative(self, order):
        K_new = self.K
        for i in range(order):
            K_new = K_new @ self.A

        derivative_polynomials = [
            poly.derivative(order) for poly in self.polynomial_parts
        ]
        # Derivative for exponential part needs special handling if it involves differentiation w.r.t time
        return ExponentialPlusPiecewisePolynomial(
            K_new, self.A, self.alpha, derivative_polynomials
        )


class ZmpPlanner:
    def __init__(self):
        self.planned_ = False
        self.kStationaryThreshold = 1e-6

    def control_com_acc(self, time, x):
        if not self.planned_:
            raise ValueError("Plan must be called first.")

        # Eq. 20 in [1]
        yf = self.zmp_ref_traj[-1]
        x_bar = x.copy()
        x_bar[:2] -= yf
        return self.K_ @ x_bar + self.k2.value(time)

    def plan(self, zmp_ref_traj, height, gravity, Qy, R, plan_t_step):
        zmp_d_degree = 1

        n_segments = len(zmp_ref_traj) - 1
        self.zmp_ref_traj = zmp_ref_traj
        self.Qy_ = Qy
        self.R_ = R

        # Eq. 1 and 2 in [1]
        A = np.zeros((4, 4))
        A[2:, 2:] = np.eye(2)
        B = np.zeros((4, 2))
        B[2:, :] = np.eye(2)
        C = np.zeros((2, 4))
        C[:, :2] = np.eye(2)
        D = -height / gravity * np.eye(2)

        # Eq. 9 - 14 in [1]
        Q1 = C.T @ Qy @ C
        R1 = R + D.T @ Qy @ D
        N = C.T @ Qy @ D
        R1_inv = np.linalg.inv(R1)

        K, S, _ = control.lqr(A, B, Q1, R1, N)
        self.K = -K

        # Computes the time varying linear and constant term in the value function
        # and linear policy. Also known as the backward pass.
        NB = N.T + B.T @ S
        # Eq. 23, 24 in [1]
        A2 = NB.T @ R1_inv @ B.T - A.T
        B2 = 2 * (C.T - NB.T @ R1_inv @ D) @ Qy
        A2_inv = np.linalg.inv(A2)

        # Last desired ZMP
        zmp_ref_last = zmp_ref_traj[-1]
        tmp4 = np.zeros(4)

        alpha = np.zeros((4, n_segments))
        beta = [np.zeros((4, zmp_d_degree + 1)) for _ in range(n_segments)]
        gamma = [np.zeros((2, zmp_d_degree + 1)) for _ in range(n_segments)]
        c = [np.zeros((2, zmp_d_degree + 1)) for _ in range(n_segments)]

        beta_poly = [None] * n_segments
        gamma_poly = [None] * n_segments

        delta_time_vec = np.zeros(zmp_d_degree + 1)
        delta_time_vec[0] = 1

        # Algorithm 1 in [1] to solve for parameters of s2 and k2
        for t in range(n_segments - 1, -1, -1):
            x1, y1 = zmp_ref_traj[t]
            x2, y2 = zmp_ref_traj[t + 1]

            c[t][:] = 0
            c[t][:, : zmp_d_degree + 1] = np.array([[x1, x2 - x1], [y1, y2 - y1]])
            # switch to zbar coord
            c[t][:, 0] -= zmp_ref_last

            # degree 4
            beta[t][:, zmp_d_degree] = -A2_inv @ B2 @ c[t][:, zmp_d_degree]
            gamma[t][:, zmp_d_degree] = (
                R1_inv @ D @ Qy @ c[t][:, zmp_d_degree]
                - 0.5 * R1_inv @ B.T @ beta[t][:, zmp_d_degree]
            )

            for d in range(zmp_d_degree - 1, -1, -1):
                beta[t][:, d] = A2_inv @ ((d + 1) * beta[t][:, d + 1] - B2 @ c[t][:, d])
                gamma[t][:, d] = (
                    R1_inv @ D @ Qy @ c[t][:, d] - 0.5 * R1_inv @ B.T @ beta[t][:, d]
                )

            if t == n_segments - 1:
                tmp4 = np.zeros(4)
            else:
                tmp4 = alpha[:, t + 1] + beta[t + 1][:, 0]

            dt = zmp_ref_traj.duration(t)
            A2exp = expm(A2 * dt)
            for i in range(zmp_d_degree + 1):
                delta_time_vec[i] = dt**i
            tmp4 = tmp4 - beta[t] @ delta_time_vec

            alpha[:, t] = np.linalg.inv(A2exp) @ tmp4

            beta_poly[t] = [PPoly.from_coeffs(beta[t][n, :]) for n in range(4)]
            gamma_poly[t] = [PPoly.from_coeffs(gamma[t][n, :]) for n in range(2)]

        # Eq. 25 in [1]
        beta_traj = PPoly.from_splines(beta_poly, plan_t_step * np.arange(n_segments))
        self.s2 = ExponentialPlusPiecewisePolynomial(np.eye(4), A2, alpha, beta_traj)

        # Eq. 28 in [1]
        gamma_traj = PPoly.from_splines(gamma_poly, zmp_ref_traj.get_segment_times())
        self.k2 = ExponentialPlusPiecewisePolynomial(
            -0.5 * R1_inv @ B.T, A2, alpha, gamma_traj
        )

        # # Computes the nominal CoM trajectory. Also known as the forward pass.
        # # Eq. 35, 36 in [1]
        # Az = np.zeros((8, 8))
        # Az[:4, :4] = A + B @ self.K_
        # Az[:4, 4:] = -0.5 * B @ self.R1i_ @ B.T
        # Az[4:, 4:] = A2
        # Azi = np.linalg.inv(Az)
        # Bz = np.zeros((8, 2))
        # Bz[:4, :] = B @ self.R1i_ @ D @ Qy
        # Bz[4:, :] = B2

        # a = np.zeros((8, n_segments))
        # a[4:, :] = alpha
        # b_poly = [None] * n_segments

        # b = [np.zeros((4, zmp_d_degree + 1)) for _ in range(n_segments)]
        # tmp81 = np.zeros(8)
        # I48 = np.zeros((4, 8))
        # I48[:, :4] = np.eye(4)

        # x = x0.copy()
        # x[:2] -= zmp_ref_last

        # # Algorithm 2 in [1] to solve for the CoM trajectory
        # for t in range(n_segments):
        #     dt = plan_t_step
        #     b[t][:, zmp_d_degree] = -Azi[:4, :] @ Bz @ c[t][:, zmp_d_degree]
        #     for d in range(zmp_d_degree - 1, -1, -1):
        #         tmp81[:4] = b[t][:, d + 1]
        #         tmp81[4:] = beta[t][:, d + 1]
        #         tmp81 *= d + 1
        #         b[t][:, d] = Azi[:4, :] @ (tmp81 - Bz @ c[t][:, d])

        #     a[:4, t] = x - b[t][:, 0]

        #     Az_exp = expm(Az * dt)
        #     for i in range(zmp_d_degree + 1):
        #         delta_time_vec[i] = dt**i
        #     x = I48 @ Az_exp @ a[:, t] + b[t] @ delta_time_vec

        #     b[t][:2, 0] += zmp_ref_last  # Map CoM position back to world frame

        #     b_poly[t] = [PPoly.from_coeffs(b[t][n, :]) for n in range(2)]

        # tmp28 = np.zeros((2, 8))
        # tmp28[:, :2] = np.eye(2)
        # b_traj = PPoly.from_splines(b_poly, zmp_ref_traj.get_segment_times())

        # Com_ = ExponentialPlusPiecewisePolynomial(tmp28, Az, a, b_traj)
        # Comd_ = Com_.derivative()
        # Comdd_ = Comd_.derivative()

        self.planned_ = True
