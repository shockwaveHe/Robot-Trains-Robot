from typing import List

import control  # type: ignore
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from toddlerbot.utils.constants import GRAVITY


class PPoly:
    def __init__(self, c: jax.Array, x: jax.Array):
        """
        x: breakpoints (knots) of the piecewise polynomial.
        c: coefficients of the polynomials for each interval. The shape should be (k, n), where
            k is the degree of the polynomial + 1, and n is the number of intervals.
        """
        self.c = c
        self.x = x

    def __call__(self, t: float | jax.Array):
        # Find the index of the interval that t falls into
        idx = jnp.clip(  # type: ignore
            jnp.searchsorted(self.x, t, side="right") - 1,  # type: ignore
            0,
            len(self.x) - 2,
        )

        # Calculate the polynomial value using Horner's method
        dt = t - self.x[idx]

        c2 = self.c.copy()
        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = jnp.zeros((1,) + c2.shape[1:], dtype=c2.dtype)  # type: ignore

        # Evaluate the polynomial using Horner's method
        result = c2[0, idx, :]
        for i in range(1, c2.shape[0]):
            result = result * dt[..., None] + c2[i, idx, :]

        return result

    def derivative(self, order: int = 1):  # type: ignore
        """
        Returns a new JAXPiecewisePolynomial representing the derivative of this polynomial.
        """
        if order == 0:
            return self

        new_c = self.c[:-1] * jnp.arange(self.c.shape[0] - 1, 0, -1)[:, None, None]  # type: ignore

        return PPoly(new_c, self.x).derivative(order - 1)  # type: ignore


class ExpPlusPPoly:
    def __init__(
        self,
        K: jax.Array,
        A: jax.Array,
        alpha: jax.Array,
        ppoly: PPoly,
    ):
        self.K = K
        self.A = A
        self.alpha = alpha
        self.ppoly = ppoly

    def value(self, t: float | jax.Array) -> jax.Array:
        # Evaluate the polynomial part at time t
        result = self.ppoly(t)

        # Determine the index of the segment that contains the time t
        segment_index = jnp.searchsorted(self.ppoly.x, t, side="right") - 1  # type: ignore
        # Ensure the index is within the valid range
        segment_index = max(0, min(segment_index, len(self.ppoly.x) - 2))  # type: ignore
        # Calculate the time offset from the beginning of the segment
        tj = self.ppoly.x[segment_index]
        # Compute the exponential part
        exponential = expm(self.A * (t - tj))  # type: ignore
        # Compute result combining the polynomial and exponential parts
        result += (  # type: ignore
            self.K @ exponential @ self.alpha[:, segment_index : segment_index + 1]
        ).flatten()  # type: ignore

        return result  # type: ignore

    def derivative(self, order: int):
        K_new = self.K
        for _ in range(order):
            K_new = K_new @ self.A

        # Derivative for exponential part needs special handling if it involves differentiation w.r.t time
        return ExpPlusPPoly(
            K_new,
            self.A,
            self.alpha,
            self.ppoly.derivative(order),  # type: ignore
        )


class ZMPPlanner:
    def __init__(self):
        self.planned = False

    def plan(
        self,
        time_steps: jax.Array,
        zmp_d: List[jax.Array],
        x0: jax.Array,
        com_z: float,
        Qy: jax.Array,
        R: jax.Array,
    ):
        self.time_steps = time_steps
        self.zmp_d = zmp_d

        # Eq. 1 and 2 in [1]
        A = jnp.zeros((4, 4), dtype=jnp.float32)  # type: ignore
        A = A.at[:2, 2:].set(jnp.eye(2, dtype=jnp.float32))  # type: ignore
        B = jnp.zeros((4, 2), dtype=jnp.float32)  # type: ignore
        B = B.at[2:, :].set(jnp.eye(2, dtype=jnp.float32))  # type: ignore
        C = jnp.zeros((2, 4), dtype=jnp.float32)  # type: ignore
        self.C = C.at[:, :2].set(jnp.eye(2, dtype=jnp.float32))  # type: ignore
        self.D = -com_z / GRAVITY * jnp.eye(2, dtype=jnp.float32)  # type: ignore

        # Eq. 9 - 14 in [1]
        Q1 = self.C.T @ Qy @ self.C
        R1 = R + self.D.T @ Qy @ self.D
        N = self.C.T @ Qy @ self.D
        R1_inv = jnp.linalg.inv(R1)  # type: ignore

        K, S, _ = control.lqr(A, B, Q1, R1, N)  # type: ignore
        self.K = -K

        # Computes the time varying linear and constant term in the value function
        # and linear policy. Also known as the backward pass.
        NB = N.T + B.T @ S
        # Eq. 23, 24 in [1]
        A2 = NB.T @ R1_inv @ B.T - A.T  # type: ignore
        B2 = 2 * (self.C.T - NB.T @ R1_inv @ self.D) @ Qy  # type: ignore
        A2_inv = jnp.linalg.inv(A2)  # type: ignore

        # Last desired ZMP
        zmp_ref_last = zmp_d[-1]
        vec4 = jnp.zeros(4, dtype=jnp.float32)  # type: ignore

        n_segments = len(zmp_d) - 1
        alpha = jnp.zeros((4, n_segments), dtype=jnp.float32)  # type: ignore
        beta = [jnp.zeros((4, 1), dtype=jnp.float32) for _ in range(n_segments)]  # type: ignore
        gamma = [jnp.zeros((2, 1), dtype=jnp.float32) for _ in range(n_segments)]  # type: ignore
        c = [jnp.zeros((2, 1), dtype=jnp.float32) for _ in range(n_segments)]  # type: ignore

        # Algorithm 1 in [1] to solve for parameters of s2 and k2
        for t in range(n_segments - 1, -1, -1):
            # Assume linear interpolation between zmp points
            c[t] = c[t].at[:, 0].set(zmp_d[t] - zmp_ref_last)  # type: ignore

            # degree 4
            beta[t] = beta[t].at[:, 0].set(-A2_inv @ B2 @ c[t][:, 0])  # type: ignore
            gamma_new = (  # type: ignore
                R1_inv @ self.D @ Qy @ c[t][:, 0] - 0.5 * R1_inv @ B.T @ beta[t][:, 0]
            )
            gamma[t] = gamma[t].at[:, 0].set(gamma_new)  # type: ignore

            dt = time_steps[t + 1] - time_steps[t]
            A2exp = expm(A2 * dt)  # type: ignore

            if t == n_segments - 1:
                vec4 = -beta[t]
            else:
                vec4 = alpha[:, t + 1 : t + 2] + beta[t + 1] - beta[t]

            alpha = alpha.at[:, t].set((jnp.linalg.inv(A2exp) @ vec4).squeeze())  # type: ignore

        # (degree+1, num_vars, num_segments)
        all_beta_coeffs = jnp.transpose(  # type: ignore
            jnp.stack(beta, axis=1),  # type: ignore
            (2, 1, 0),
        )
        all_gamma_coeffs = jnp.transpose(  # type: ignore
            jnp.stack(gamma, axis=1),  # type: ignore
            (2, 1, 0),
        )

        # Eq. 25 in [1]
        beta_traj = PPoly(all_beta_coeffs, time_steps)
        self.s2 = ExpPlusPPoly(
            jnp.eye(4, dtype=jnp.float32),  # type: ignore
            A2,  # type: ignore
            alpha,
            beta_traj,
        )

        # Eq. 28 in [1]
        gamma_traj = PPoly(all_gamma_coeffs, time_steps)
        self.k2 = ExpPlusPPoly(
            -0.5 * R1_inv @ B.T,  # type: ignore
            A2,  # type: ignore
            alpha,
            gamma_traj,
        )

        # Computes the nominal CoM trajectory. Also known as the forward pass.
        # Eq. 35, 36 in [1]
        Az = jnp.zeros((8, 8), dtype=jnp.float32)  # type: ignore
        Az = Az.at[:4, :4].set(A + B @ self.K)  # type: ignore
        Az = Az.at[:4, 4:].set(-0.5 * B @ R1_inv @ B.T)  # type: ignore
        Az = Az.at[4:, 4:].set(A2)  # type: ignore
        Azi = jnp.linalg.inv(Az)  # type: ignore
        Bz = jnp.zeros((8, 2), dtype=jnp.float32)  # type: ignore
        Bz = Bz.at[:4, :].set(B @ R1_inv @ self.D @ Qy)  # type: ignore
        Bz = Bz.at[4:, :].set(B2)  # type: ignore

        a = jnp.zeros((8, n_segments), dtype=jnp.float32)  # type: ignore
        a = a.at[4:, :].set(alpha)  # type: ignore

        b = [jnp.zeros((4, 1), dtype=jnp.float32) for _ in range(n_segments)]  # type: ignore
        I48 = jnp.zeros((4, 8), dtype=jnp.float32)  # type: ignore
        I48 = I48.at[:, :4].set(jnp.eye(4, dtype=jnp.float32))  # type: ignore

        x = x0.copy()
        x = x.at[:2].add(-zmp_ref_last)

        # Algorithm 2 in [1] to solve for the CoM trajectory
        for t in range(n_segments):
            dt = time_steps[t + 1] - time_steps[t]
            b[t] = b[t].at[:, 0].set(-Azi[:4, :] @ Bz @ c[t][:, 0])  # type: ignore
            a = a.at[:4, t].set(x - b[t][:, 0])  # type: ignore
            Az_exp = expm(Az * dt)  # type: ignore
            x = I48 @ Az_exp @ a[:, t] + b[t].squeeze()  # type: ignore
            b[t] = b[t].at[:2, 0].add(zmp_ref_last)

        mat28 = jnp.zeros((2, 8), dtype=jnp.float32)  # type: ignore
        mat28 = mat28.at[:, :2].set(jnp.eye(2, dtype=jnp.float32))  # type: ignore
        all_b_coeffs = jnp.transpose(jnp.stack(b, axis=1), (2, 1, 0))  # type: ignore
        b_traj = PPoly(all_b_coeffs[..., :2], time_steps)

        self.com_pos = ExpPlusPPoly(mat28, Az, a, b_traj)
        self.com_vel = self.com_pos.derivative(1)
        self.com_acc = self.com_vel.derivative(1)

        self.planned = True

    def get_optim_com_acc(self, time: float | jax.Array, x: jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        # Eq. 20 in [1]
        yf = self.zmp_d[-1]
        x_bar = x.copy()
        x_bar = x_bar.at[:2].add(-yf)
        return self.K @ x_bar + self.k2.value(time)

    def com_acc_to_cop(self, x: jax.Array, u: jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")
        return self.C @ x + self.D @ u

    def get_desired_zmp_traj(self):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.time_steps, self.zmp_d

    def get_desired_zmp(self, time: float | jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.zmp_d[jnp.searchsorted(self.time_steps, time, side="right") - 1]  # type: ignore

    def get_nominal_com(self, time: float | jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_pos.value(time)

    def get_nominal_com_vel(self, time: float | jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_vel.value(time)

    def get_nominal_com_acc(self, time: float | jax.Array):
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_acc.value(time)
