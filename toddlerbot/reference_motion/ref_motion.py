import numpy as np


class KinematicMotionGenerator:
    def __init__(self, initial_state, motion_type="perpetual"):
        if motion_type not in ["perpetual", "periodic", "episodic"]:
            raise ValueError(
                "motion_type must be 'perpetual', 'periodic', or 'episodic'"
            )

        self.state = initial_state
        self.motion_type = motion_type

    def update_state(self, p_t, theta_t, v_t, omega_t, q_t, q_dot_t, c_L_t, c_R_t):
        self.state = {
            "position": p_t,
            "orientation": theta_t,
            "linear_velocity": v_t,
            "angular_velocity": omega_t,
            "joint_positions": q_t,
            "joint_velocities": q_dot_t,
            "left_contact": c_L_t,
            "right_contact": c_R_t,
        }

    def generate_motion(self, f_t, phi_t=None, g_t=None):
        if self.motion_type == "perpetual":
            return self._f_perp(f_t, g_t)
        elif self.motion_type == "periodic":
            return self._f_peri(f_t, phi_t, g_t)
        elif self.motion_type == "episodic":
            return self._f_epis(f_t, phi_t)

    def _f_perp(self, f_t, g_perp_t):
        if g_perp_t is None:
            raise ValueError("g_perp_t is required for perpetual motion")
        return f_t + g_perp_t

    def _f_peri(self, f_t, phi_t, g_peri_t):
        if phi_t is None or g_peri_t is None:
            raise ValueError("phi_t and g_peri_t are required for periodic motion")
        x_t = f_t + np.sin(phi_t) * g_peri_t
        phi_dot_t = np.cos(phi_t) * g_peri_t
        return x_t, phi_dot_t

    def _f_epis(self, f_t, phi_t):
        if phi_t is None:
            raise ValueError("phi_t is required for episodic motion")
        return f_t * phi_t
