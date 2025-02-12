from typing import List

import control

from toddlerbot.utils.array_utils import ArrayType, expm, inplace_add, inplace_update
from toddlerbot.utils.array_utils import array_lib as np

GRAVITY = 9.81


class PPoly:
    """Piecewise polynomial class."""

    def __init__(self, c: ArrayType, x: ArrayType):
        """Initializes the piecewise polynomial with given breakpoints and coefficients.

        Args:
            c (ArrayType): Coefficients of the polynomials for each interval. The shape should be (k, n), where k is the degree of the polynomial + 1, and n is the number of intervals.
            x (ArrayType): Breakpoints (knots) of the piecewise polynomial.
        """
        self.c = c
        self.x = x

    def __call__(self, t: float | ArrayType) -> ArrayType:
        """Returns the value of the polynomial and exponential components at the specified time `t`.

        Args:
            t (float | ArrayType): The time or array of times at which to evaluate the function.

        Returns:
            ArrayType: The evaluated value of the polynomial and exponential components at time `t`.
        """
        # Find the index of the interval that t falls into
        idx = np.clip(np.searchsorted(self.x, t, side="right") - 1, 0, len(self.x) - 2)

        # Calculate the polynomial value using Horner's method
        dt = t - self.x[idx]

        c2 = self.c.copy()
        if c2.shape[0] == 0:
            # Derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # Evaluate the polynomial using Horner's method
        result = c2[0, idx, :]
        for i in range(1, c2.shape[0]):
            result = result * dt[..., None] + c2[i, idx, :]  # type: ignore

        return result

    def derivative(self, order: int = 1) -> "PPoly":
        """Compute the derivative of the piecewise polynomial.

        Args:
            order (int): The order of the derivative to compute. Defaults to 1.

        Returns:
            PPoly: A new piecewise polynomial representing the derivative of the specified order.
        """
        if order == 0:
            return self

        new_c = self.c[:-1] * np.arange(self.c.shape[0] - 1, 0, -1)[:, None, None]

        return PPoly(new_c, self.x).derivative(order - 1)


class ExpPlusPPoly:
    """Exponential plus piecewise polynomial class."""

    def __init__(
        self,
        K: ArrayType,
        A: ArrayType,
        alpha: ArrayType,
        ppoly: PPoly,
    ):
        """Initializes the object with given parameters.

        Args:
            K (ArrayType): An array representing the stiffness matrix.
            A (ArrayType): An array representing the area matrix.
            alpha (ArrayType): An array representing the thermal expansion coefficients.
            ppoly (PPoly): A piecewise polynomial object for interpolation.
        """
        self.K = K
        self.A = A
        self.alpha = alpha
        self.ppoly = ppoly

    def value(self, t: float | ArrayType) -> ArrayType:
        """Evaluate the value of a piecewise polynomial with an exponential component at a given time.

        Args:
            t (float | ArrayType): The time or array of times at which to evaluate the function.

        Returns:
            ArrayType: The evaluated value(s) of the function at the specified time(s).
        """
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
        # Compute the result by combining the polynomial and exponential parts
        result += (
            self.K @ exponential @ self.alpha[:, segment_index : segment_index + 1]
        ).flatten()

        return result

    def derivative(self, order: int) -> "ExpPlusPPoly":
        """Computes the derivative of the ExpPlusPPoly object to a specified order.

        Args:
            order (int): The order of the derivative to compute.

        Returns:
            ExpPlusPPoly: A new ExpPlusPPoly object representing the derivative of the specified order.
        """
        K_new = self.K
        for _ in range(order):
            K_new = K_new @ self.A

        # Derivative for the exponential part needs special handling if it involves differentiation w.r.t time
        return ExpPlusPPoly(
            K_new,
            self.A,
            self.alpha,
            self.ppoly.derivative(order),
        )


class ZMPPlanner:
    """Zero Moment Point (ZMP) planner class based on the paper "A Generalized ZMP Preview Control for Bipedal Walking" by Kajita et al."""

    def __init__(self):
        """Initializes an instance of the class with a default planned status.

        Attributes:
            planned (bool): Indicates whether the instance is planned. Defaults to False.
        """
        self.planned = False

    def plan(
        self,
        time_steps: ArrayType,
        zmp_d: List[ArrayType],
        x0: ArrayType,
        com_z: float,
        Qy: ArrayType,
        R: ArrayType,
    ):
        """Plans the Center of Mass (CoM) trajectory and control gains for a system based on the desired Zero Moment Point (ZMP) trajectory.

        Args:
            time_steps (ArrayType): Array of time steps for the trajectory.
            zmp_d (List[ArrayType]): List of desired ZMP positions at each time step.
            x0 (ArrayType): Initial state of the system.
            com_z (float): Height of the Center of Mass.
            Qy (ArrayType): Weighting matrix for the output in the cost function.
            R (ArrayType): Weighting matrix for the control input in the cost function.

        This function computes the time-varying linear and constant terms in the value function, solves for parameters of the CoM trajectory, and updates the planned CoM position, velocity, and acceleration.
        """
        self.time_steps = time_steps
        self.zmp_d = zmp_d

        # Eq. 1 and 2 in [1]
        A = np.zeros((4, 4), dtype=np.float32)
        A = inplace_update(
            A,
            (slice(None, 2), slice(2, None)),
            np.eye(2, dtype=np.float32),
        )
        B = np.zeros((4, 2), dtype=np.float32)
        B = inplace_update(
            B,
            (slice(2, None), slice(None)),
            np.eye(2, dtype=np.float32),
        )
        C = np.zeros((2, 4), dtype=np.float32)
        self.C = inplace_update(
            C,
            (slice(None), slice(None, 2)),
            np.eye(2, dtype=np.float32),
        )
        self.D = -com_z / GRAVITY * np.eye(2, dtype=np.float32)

        # Eq. 9 - 14 in [1]
        Q1 = self.C.T @ Qy @ self.C
        R1 = R + self.D.T @ Qy @ self.D
        N = self.C.T @ Qy @ self.D
        R1_inv = np.linalg.inv(R1)

        K, S, _ = control.lqr(A, B, Q1, R1, N)
        self.K = -K

        # Computes the time-varying linear and constant terms in the value function
        NB = N.T + B.T @ S
        A2 = NB.T @ R1_inv @ B.T - A.T
        B2 = 2 * (self.C.T - NB.T @ R1_inv @ self.D) @ Qy
        A2_inv = np.linalg.inv(A2)

        # Last desired ZMP
        zmp_ref_last = zmp_d[-1]
        vec4 = np.zeros(4, dtype=np.float32)

        n_segments = len(zmp_d) - 1
        alpha = np.zeros((4, n_segments), dtype=np.float32)
        beta = [np.zeros((4, 1), dtype=np.float32) for _ in range(n_segments)]
        gamma = [np.zeros((2, 1), dtype=np.float32) for _ in range(n_segments)]
        c = [np.zeros((2, 1), dtype=np.float32) for _ in range(n_segments)]

        # Algorithm 1 in [1] to solve for parameters of s2 and k2
        for t in range(n_segments - 1, -1, -1):
            # Assume linear interpolation between ZMP points
            c[t] = inplace_update(c[t], (slice(None), 0), zmp_d[t] - zmp_ref_last)

            # degree 4
            beta[t] = inplace_update(
                beta[t],
                (slice(None), 0),
                -A2_inv @ B2 @ c[t][:, 0],
            )
            gamma_new = (
                R1_inv @ self.D @ Qy @ c[t][:, 0] - 0.5 * R1_inv @ B.T @ beta[t][:, 0]
            )
            gamma[t] = inplace_update(gamma[t], (slice(None), 0), gamma_new)

            dt = time_steps[t + 1] - time_steps[t]
            A2exp = expm(A2 * dt)

            if t == n_segments - 1:
                vec4 = -beta[t]
            else:
                vec4 = alpha[:, t + 1 : t + 2] + beta[t + 1] - beta[t]

            alpha = inplace_update(
                alpha,
                (slice(None), t),
                (np.linalg.inv(A2exp) @ vec4).squeeze(),
            )

        # (degree + 1, num_vars, num_segments)
        all_beta_coeffs = np.transpose(np.stack(beta, axis=1), (2, 1, 0))
        all_gamma_coeffs = np.transpose(np.stack(gamma, axis=1), (2, 1, 0))

        # Eq. 25 in [1]
        beta_traj = PPoly(all_beta_coeffs, time_steps)
        self.s2 = ExpPlusPPoly(
            np.eye(4, dtype=np.float32),
            A2,
            alpha,
            beta_traj,
        )

        # Eq. 28 in [1]
        gamma_traj = PPoly(all_gamma_coeffs, time_steps)
        self.k2 = ExpPlusPPoly(
            -0.5 * R1_inv @ B.T,
            A2,
            alpha,
            gamma_traj,
        )

        # Computes the nominal CoM trajectory. Also known as the forward pass.
        # Eq. 35, 36 in [1]
        Az = np.zeros((8, 8), dtype=np.float32)
        Az = inplace_update(Az, (slice(None, 4), slice(None, 4)), A + B @ self.K)
        Az = inplace_update(
            Az, (slice(None, 4), slice(4, None)), -0.5 * B @ R1_inv @ B.T
        )
        Az = inplace_update(Az, (slice(4, None), slice(4, None)), A2)
        Azi = np.linalg.inv(Az)
        Bz = np.zeros((8, 2), dtype=np.float32)
        Bz = inplace_update(Bz, (slice(None, 4), slice(None)), B @ R1_inv @ self.D @ Qy)
        Bz = inplace_update(Bz, (slice(4, None), slice(None)), B2)

        a = np.zeros((8, n_segments), dtype=np.float32)
        a = inplace_update(a, (slice(4, None), slice(None)), alpha)

        b = [np.zeros((4, 1), dtype=np.float32) for _ in range(n_segments)]
        i48 = np.zeros((4, 8), dtype=np.float32)
        i48 = inplace_update(
            i48,
            (slice(None), slice(None, 4)),
            np.eye(4, dtype=np.float32),
        )

        x = x0.copy()
        x = inplace_add(x, slice(None, 2), -zmp_ref_last)

        # Algorithm 2 in [1] to solve for the CoM trajectory
        for t in range(n_segments):
            dt = time_steps[t + 1] - time_steps[t]
            b[t] = inplace_update(b[t], (slice(None), 0), -Azi[:4, :] @ Bz @ c[t][:, 0])
            a = inplace_update(a, (slice(None, 4), t), x - b[t][:, 0])
            Az_exp = expm(Az * dt)
            x = i48 @ Az_exp @ a[:, t] + b[t].squeeze()
            b[t] = inplace_add(b[t], (slice(None, 2), 0), zmp_ref_last)

        mat28 = np.zeros((2, 8), dtype=np.float32)
        mat28 = inplace_update(
            mat28,
            (slice(None), slice(None, 2)),
            np.eye(2, dtype=np.float32),
        )
        all_b_coeffs = np.transpose(np.stack(b, axis=1), (2, 1, 0))
        b_traj = PPoly(all_b_coeffs[..., :2], time_steps)

        self.com_pos = ExpPlusPPoly(mat28, Az, a, b_traj)
        self.com_vel = self.com_pos.derivative(1)
        self.com_acc = self.com_vel.derivative(1)

        self.planned = True

    def get_optim_com_acc(self, time: float | ArrayType, x: ArrayType) -> ArrayType:
        """Calculates the optimal center of mass acceleration.

        This function computes the optimal center of mass (CoM) acceleration based on the current state and time. It requires that a planning step has been executed prior to its invocation.

        Args:
            time (float | ArrayType): The current time or an array of time values.
            x (ArrayType): The current state vector.

        Returns:
            ArrayType: The calculated optimal CoM acceleration.

        Raises:
            ValueError: If the planning step has not been executed before calling this function.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        # Eq. 20 in [1]
        yf = self.zmp_d[-1]
        x_bar = x.copy()
        x_bar = inplace_add(x_bar, slice(None, 2), -yf)
        return self.K @ x_bar + self.k2.value(time)

    def com_acc_to_cop(self, x: ArrayType, u: ArrayType) -> ArrayType:
        """Calculates the control output based on the current state and input.

        This function computes the control output using the state vector `x` and the input vector `u` by applying the transformation matrices `C` and `D`. It requires that the planning step has been completed prior to execution.

        Args:
            x (ArrayType): The current state vector.
            u (ArrayType): The current input vector.

        Returns:
            ArrayType: The resulting control output vector.

        Raises:
            ValueError: If the planning step has not been completed.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")
        return self.C @ x + self.D @ u

    def get_desired_zmp_traj(self):
        """Retrieves the desired Zero Moment Point (ZMP) trajectory.

        Returns:
            tuple: A tuple containing the time steps and the desired ZMP trajectory.

        Raises:
            ValueError: If the plan has not been executed prior to calling this method.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.time_steps, self.zmp_d

    def get_desired_zmp(self, time: float | ArrayType) -> ArrayType:
        """Retrieves the desired Zero Moment Point (ZMP) at a specified time.

        Args:
            time (float | ArrayType): The time or array of times at which to retrieve the desired ZMP.

        Returns:
            ArrayType: The desired ZMP corresponding to the given time.

        Raises:
            ValueError: If the plan has not been initialized by calling the `plan` method.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.zmp_d[np.searchsorted(self.time_steps, time, side="right") - 1]

    def get_nominal_com(self, time: float | ArrayType) -> ArrayType:
        """Retrieve the nominal center of mass (CoM) position at a specified time or times.

        Args:
            time (float | ArrayType): The time or array of times at which to evaluate the CoM position.

        Returns:
            ArrayType: The nominal CoM position(s) corresponding to the given time(s).

        Raises:
            ValueError: If the plan has not been initialized by calling the `plan` method.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_pos.value(time)

    def get_nominal_com_vel(self, time: float | ArrayType) -> ArrayType:
        """Retrieve the nominal center of mass velocity at a specified time.

        Args:
            time (float | ArrayType): The time or array of times at which to evaluate the center of mass velocity.

        Returns:
            ArrayType: The nominal center of mass velocity at the specified time(s).

        Raises:
            ValueError: If the plan has not been called prior to this method.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_vel.value(time)

    def get_nominal_com_acc(self, time: float | ArrayType) -> ArrayType:
        """Calculates the nominal center of mass acceleration at a given time or times.

        Args:
            time (float | ArrayType): The time or array of times at which to evaluate the center of mass acceleration.

        Returns:
            ArrayType: The center of mass acceleration at the specified time(s).

        Raises:
            ValueError: If the plan has not been initialized by calling the `plan` method.
        """
        if not self.planned:
            raise ValueError("Plan must be called first.")

        return self.com_acc.value(time)
