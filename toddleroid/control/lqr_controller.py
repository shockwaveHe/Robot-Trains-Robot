import control
import numpy as np

from toddleroid.control import BaseController


class LQRController(BaseController):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        discrete: bool = False,
    ) -> None:
        """
        Initialize the LQR controller with the system dynamics and cost matrices.

        Args:
            A (np.ndarray): State transition matrix.
            B (np.ndarray): Control matrix.
            Q (np.ndarray): State cost matrix.
            R (np.ndarray): Control cost matrix.
            discrete (bool): Flag indicating if the system is discrete (default True).
                             If True, uses the discrete-time LQR, otherwise continuous-time.
        """
        super().__init__()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.discrete = discrete
        self.K = self._calculate_lqr_gain()

    def _calculate_lqr_gain(self) -> np.ndarray:
        """
        Calculate the LQR gain matrix based on system matrices and discrete flag.

        Returns:
            np.ndarray: The calculated LQR gain matrix.
        """
        if self.discrete:
            return control.dlqr(self.A, self.B, self.Q, self.R)[0]
        else:
            return control.lqr(self.A, self.B, self.Q, self.R)[0]

    def compute_control(
        self, desired_state: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        """
        Compute the control action using LQR.

        Args:
            desired_state (np.ndarray): The desired state of the system.
            current_state (np.ndarray): The current state of the system.

        Returns:
            np.ndarray: The control input for the system.
        """
        error = desired_state - current_state
        control_input = -self.K @ error
        return control_input

    def update_parameters(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ) -> None:
        """
        Update the LQR parameters.

        Args:
            A (np.ndarray, optional): Updated state transition matrix.
            B (np.ndarray, optional): Updated control matrix.
            Q (np.ndarray, optional): Updated state cost matrix.
            R (np.ndarray, optional): Updated control cost matrix.
        """
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        self.K = self._calculate_lqr_gain()


def test_lqr_controller() -> None:
    """
    Test function for the LQRController class.
    """
    # Define a simple test case
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)

    # Initialize LQR controller
    lqr_controller = LQRController(A, B, Q, R)

    expected_K = np.array([[1.0, 1.73205081]])
    assert np.allclose(
        lqr_controller.K, expected_K
    ), f"LQRController test failed: K matrix mismatch. Expected {expected_K}, got {lqr_controller.K}"

    current_state = np.array([0, 0])
    desired_state = np.array([1, 1])
    # Compute control
    control_input = lqr_controller.compute_control(desired_state, current_state)

    # Validate control input
    expected_control_input = np.array([-2.73205081])
    assert np.allclose(
        control_input, expected_control_input
    ), f"LQRController test failed: Control input mismatch. Expected {expected_control_input}, got {control_input}"

    print("LQRController test passed. Computed control input is as expected.")


if __name__ == "__main__":
    test_lqr_controller()
