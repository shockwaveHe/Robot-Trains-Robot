# Derived from the original code at https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024/blob/main/LIPM/LIPM_3D.py
from typing import List

from toddlerbot.utils.array_utils import array_lib as np

GRAVITY = 9.81


class LIPM3DPlanner:
    def __init__(
        self,
        control_dt: float = 0.001,
        T: float = 1.0,
        T_d: float = 0.3,
        s_d: float = 0.12,
        w_d: float = 0.035,
        support_leg: str = "left_leg",
    ) -> None:
        self.dt: float = control_dt
        self.t: float = 0.0
        self.T: float = T  # step time
        self.T_d: float = T_d  # desired step duration
        self.s_d: float = s_d  # desired step length
        self.w_d: float = w_d  # desired step width

        self.eICP_x: float = 0.0  # ICP location x
        self.eICP_y: float = 0.0  # ICP location y
        self.u_x: float = 0.0  # step location x
        self.u_y: float = 0.0  # step location y

        # COM initial state
        self.x_0: float = 0.0
        self.vx_0: float = 0.0
        self.y_0: float = 0.0
        self.vy_0: float = 0.0

        # COM real-time state
        self.x_t: float = 0.0
        self.vx_t: float = 0.0
        self.y_t: float = 0.0
        self.vy_t: float = 0.0

        self.support_leg: str = support_leg
        self.support_foot_pos: List[float] = [0.0, 0.0, 0.0]
        self.left_foot_pos: List[float] = [0.0, 0.0, 0.0]
        self.right_foot_pos: List[float] = [0.0, 0.0, 0.0]
        self.COM_pos: List[float] = [0.0, 0.0, 0.0]

    def initialize_model(
        self,
        COM_pos: List[float],
        left_foot_pos: List[float],
        right_foot_pos: List[float],
    ) -> None:
        self.COM_pos = COM_pos

        if self.support_leg == "left_leg":
            self.left_foot_pos = left_foot_pos
            self.right_foot_pos = right_foot_pos
            self.support_foot_pos = left_foot_pos
        elif self.support_leg == "right_leg":
            self.left_foot_pos = left_foot_pos
            self.right_foot_pos = right_foot_pos
            self.support_foot_pos = right_foot_pos

        self.zc: float = self.COM_pos[2]
        self.w_0: float = np.sqrt(GRAVITY / self.zc)  # type: ignore

    def step(self) -> None:
        self.t += self.dt
        t: float = self.t

        self.x_t = (
            self.x_0 * np.cosh(t * self.w_0)  # type: ignore
            + self.vx_0 * np.sinh(t * self.w_0) / self.w_0  # type: ignore
        )
        self.vx_t = self.x_0 * self.w_0 * np.sinh(t * self.w_0) + self.vx_0 * np.cosh(  # type: ignore
            t * self.w_0
        )

        self.y_t = (
            self.y_0 * np.cosh(t * self.w_0)  # type: ignore
            + self.vy_0 * np.sinh(t * self.w_0) / self.w_0  # type: ignore
        )
        self.vy_t = self.y_0 * self.w_0 * np.sinh(t * self.w_0) + self.vy_0 * np.cosh(  # type: ignore
            t * self.w_0
        )

    def calculate_Xf_Vf(self) -> tuple[float, float, float, float]:
        x_f = (
            self.x_0 * np.cosh(self.T * self.w_0)  # type: ignore
            + self.vx_0 * np.sinh(self.T * self.w_0) / self.w_0  # type: ignore
        )
        vx_f = self.x_0 * self.w_0 * np.sinh(self.T * self.w_0) + self.vx_0 * np.cosh(  # type: ignore
            self.T * self.w_0
        )

        y_f = (
            self.y_0 * np.cosh(self.T * self.w_0)  # type: ignore
            + self.vy_0 * np.sinh(self.T * self.w_0) / self.w_0  # type: ignore
        )
        vy_f = self.y_0 * self.w_0 * np.sinh(self.T * self.w_0) + self.vy_0 * np.cosh(  # type: ignore
            self.T * self.w_0
        )

        return x_f, vx_f, y_f, vy_f  # type: ignore

    def calculate_foot_location_world(self, theta: float = 0.0) -> None:
        x_f, vx_f, y_f, vy_f = self.calculate_Xf_Vf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f / self.w_0
        self.eICP_y = y_f_world + vy_f / self.w_0
        b_x = self.s_d / (np.exp(self.w_0 * self.T_d) - 1)  # type: ignore
        b_y = self.w_d / (np.exp(self.w_0 * self.T_d) + 1)  # type: ignore

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y  # type: ignore
        offset_y = np.sin(theta) * original_offset_x + np.cos(theta) * original_offset_y  # type: ignore

        self.u_x = self.eICP_x + offset_x  # type: ignore
        self.u_y = self.eICP_y + offset_y  # type: ignore

    def calculate_foot_location_base(self, theta: float = 0.0) -> None:
        x_f, vx_f, y_f, vy_f = self.calculate_Xf_Vf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f / self.w_0
        self.eICP_y = y_f_world + vy_f / self.w_0
        b_x = self.s_d / (np.exp(self.w_0 * self.T_d) - 1)  # type: ignore
        b_y = self.w_d / (np.exp(self.w_0 * self.T_d) + 1)  # type: ignore

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y  # type: ignore
        offset_y = np.sin(theta) * original_offset_x - np.cos(theta) * original_offset_y  # type: ignore

        self.u_x = self.eICP_x + offset_x  # type: ignore
        self.u_y = self.eICP_y + offset_y  # type: ignore

    def switch_support_leg(self) -> None:
        if self.support_leg == "left_leg":
            print("\n---- switch the support leg to the right leg")
            self.support_leg = "right_leg"
            COM_pos_x = self.x_t + self.left_foot_pos[0]
            COM_pos_y = self.y_t + self.left_foot_pos[1]
            self.x_0 = COM_pos_x - self.right_foot_pos[0]
            self.y_0 = COM_pos_y - self.right_foot_pos[1]
            self.support_foot_pos = self.right_foot_pos
        elif self.support_leg == "right_leg":
            print("\n---- switch the support leg to the left leg")
            self.support_leg = "left_leg"
            COM_pos_x = self.x_t + self.right_foot_pos[0]
            COM_pos_y = self.y_t + self.right_foot_pos[1]
            self.x_0 = COM_pos_x - self.left_foot_pos[0]
            self.y_0 = COM_pos_y - self.left_foot_pos[1]
            self.support_foot_pos = self.left_foot_pos

        self.t = 0
        self.vx_0 = self.vx_t
        self.vy_0 = self.vy_t
