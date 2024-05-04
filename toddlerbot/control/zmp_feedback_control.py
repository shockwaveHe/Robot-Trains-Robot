import control
import numpy as np

from toddlerbot.utils.constants import GRAVITY


class ZMPFeedbackController:
    def __init__(self, com_height, dt, Q_val, R_val):
        self.com_height = com_height  # Center of Mass height
        self.dt = dt  # Sampling time for discretization
        self.Q = Q_val * np.eye(4)
        self.R = R_val * np.eye(2)
        self._setup()

    def _setup(self):
        # System matrices
        # Equation 2 in https://groups.csail.mit.edu/robotics-center/public_papers/Tedrake15.pdf
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        D = -self.com_height / GRAVITY * np.eye(2)

        # Create the continuous-time state-space model
        dsys = control.ss(A, B, C, D, self.dt)
        # Compute the discrete-time LQR controller gain
        self.K, _, _ = control.dlqr(dsys, self.Q, self.R)

    def compute_zmp(self, qpos, qvel):
        self.last_qvel = qvel
        self.last_qpos = qpos
        # Here you would compute the COM and ZMP based on joint states
        # Placeholder for actual computation
        # com_acc = compute_com_acceleration(qpos, qvel, joint_acc)
        # zmp = compute_zmp(com_acc)
        # This is a simplification for demonstration

    def control(self, zmp, zmp_ref):
        x = np.concatenate([zmp - zmp_ref])
        u = -self.K @ x  # Control input considering deviation from reference
        return u


# Usage
com_height = 0.5
dt = 0.01
Q_val = 1.0
R_val = 1.0
zmp_controller = ZMPFeedbackController(com_height, dt, Q_val, R_val)
zmp = np.array([0.0, 0.0])
zmp_ref = np.array([0.0, 0.0])
control_input = zmp_controller.control(zmp, zmp_ref)
print(control_input)
