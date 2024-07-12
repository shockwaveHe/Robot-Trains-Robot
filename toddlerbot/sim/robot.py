import copy
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator  # type: ignore
from yourdfpy import URDF, Joint  # type: ignore

from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import log


class Robot:
    """Class representing a humanoid robot."""

    def __init__(self, robot_name: str):
        """
        Initialize a humanoid robot with the given name.

        Args:
            robot_name (str): The name of the robot.

        Raises:
            ValueError: If the robot name is not supported.
        """
        self.id = 0
        self.name = robot_name

        root_path = os.path.join("toddlerbot", "robot_descriptions", self.name)
        self.config_file_path = os.path.join(root_path, "config.json")
        self.cache_file_path = os.path.join(root_path, f"{self.name}_data.pkl")

        self.load_robot_config()
        self.load_robot_data()

        self.init_joint_angles = self.initialize_joint_angles()

    def load_robot_config(self):
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path, "r") as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")

    def write_robot_config(self):
        with open(self.config_file_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def load_robot_data(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "rb") as f:
                self.data: Dict[str, Any] = pickle.load(f)
                log("Loaded cached data.", header="Robot")

            # TODO: bring these fields back for the humanoid
            # self.com = robot_data["com"]
            # self.foot_size = robot_data["foot_size"]
            # self.offsets = robot_data["offsets"]
            # points, values = robot_data["ankle_fk_lookup_table"]
            # self.ankle_fk_lookup_table = LinearNDInterpolator(points, values)

        else:
            urdf_path = find_robot_file_path(self.name)
            urdf: URDF = URDF.load(urdf_path)  # type: ignore
            self.data = self.get_data(urdf)

            with open(self.cache_file_path, "wb") as f:
                pickle.dump(self.data, f)
                log("Computed and cached new data.", header="Robot")

    def get_data(self, urdf: URDF) -> Dict[str, Any]:
        data_dict: Dict[str, Any] = {}
        # if self.config.com is None:
        #     self.com = urdf.scene.center_mass
        # else:
        #     self.com = self.config.com

        # if self.config.foot_size is None:
        #     self.foot_size = self.compute_foot_size(urdf)
        # else:
        #     self.foot_size = self.config.foot_size

        # if self.config.offsets is None:
        #     self.offsets = self.compute_offsets(urdf)
        # else:
        #     self.offsets = self.config.offsets

        # points, values = self.precompute_ankle_fk_lookup()
        # self.ankle_fk_lookup_table = LinearNDInterpolator(points, values)

        return data_dict

    def compute_foot_size(self, urdf: URDF) -> np.ndarray:  # type: ignore
        foot_bounds = urdf.scene.geometry.get("left_ank_roll_link_visual.stl").bounds
        foot_ori = urdf.scene.graph.get("ank_roll_link")[0][:3, :3]
        foot_bounds_rotated = foot_bounds @ foot_ori.T
        foot_size = np.abs(foot_bounds_rotated[1] - foot_bounds_rotated[0])

        # 0.004 is the thickness of the foot pad
        return np.array([foot_size[0], foot_size[1], 0.004])

    def compute_offsets(self, urdf: URDF):
        graph = urdf.scene.graph

        offsets = {}
        # from the hip roll joint to the hip pitch joint
        offsets["z_offset_hip_roll_to_pitch"] = (
            graph.get("hip_roll_link")[0][2, 3]
            - graph.get("left_hip_pitch_link")[0][2, 3]
        )
        # from the hip pitch joint to the knee joint
        offsets["z_offset_thigh"] = (
            graph.get("left_hip_pitch_link")[0][2, 3]
            - graph.get("left_calf_link")[0][2, 3]
        )
        # the knee joint offset
        offsets["z_offset_knee"] = 0.0
        # from the knee joint to the ankle roll joint
        offsets["z_offset_shin"] = (
            graph.get("left_calf_link")[0][2, 3] - graph.get("ank_roll_link")[0][2, 3]
        )
        # from the hip center to the foot
        offsets["y_offset_com_to_foot"] = graph.get("ank_roll_link")[0][1, 3]

        # Below are for the ankle IK
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.
        ank_origin = np.array(
            [
                graph.get("ank_pitch_link")[0][0, 3],
                *graph.get("ank_roll_link")[0][1:3, 3],
            ]
        )
        offsets["s1"] = graph.get("ball_joint_ball")[0][:3, 3] - ank_origin
        offsets["f1E"] = graph.get("ank_rr_link")[0][:3, 3] - ank_origin
        offsets["nE"] = np.array([1, 0, 0])
        offsets["r"] = np.linalg.norm(
            graph.get("ank_rr_link")[0][:3, 3] - graph.get("12lf_rod_end")[0][:3, 3]
        )

        # Hard Code: Measured on the real robot (m)
        offsets["mighty_zap_len"] = 0.076

        return offsets

    def initialize_joint_angles(self) -> Dict[str, float]:
        joint_angles: Dict[str, float] = {}
        for joint_name, joint_config in self.config["joints"].items():
            joint_angles[joint_name] = joint_config["default_pos"]

        return joint_angles

    def get_joint_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str = "name",
        group: str = "all",
    ) -> List[Any]:
        attrs: List[Any] = []
        for joint_name, joint_config in self.config["joints"].items():
            if (
                key_name in joint_config
                and joint_config[key_name] == key_value
                and (joint_config["group"] == group or group == "all")
            ):
                if attr_name == "name":
                    attrs.append(joint_name)
                else:
                    attrs.append(joint_config[attr_name])

        return attrs

    def set_joint_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str,
        attr_values: Any,
        group: str = "all",
    ):
        i = 0
        for joint_name, joint_config in self.config["joints"].items():
            if key_name in joint_config in joint_config[key_name] == key_value and (
                joint_config["group"] == group or group == "all"
            ):
                if isinstance(attr_values, dict):
                    id = joint_config["id"]
                    self.config[joint_name][attr_name] = attr_values[id]
                else:
                    self.config[joint_name][attr_name] = attr_values[i]
                    i += 1

    def get_ankle_pos(
        self, joint_angles: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        left_ankle_pos: Dict[str, float] = {}
        right_ankle_pos: Dict[str, float] = {}
        for k in self.get_joint_attrs("is_closed_loop", True):
            if "left_ank" in k:
                left_ankle_pos[k] = joint_angles[k]
            if "right_ank" in k:
                right_ankle_pos[k] = joint_angles[k]
        return left_ankle_pos, right_ankle_pos

    def ankle_ik(self, ankle_pos: Dict[str, float]) -> Dict[str, float]:
        # Extracting offset values and converting to NumPy arrays
        # offsets = self.offsets
        # s1 = np.array(offsets["s1"])
        # s2 = np.array([s1[0], -s1[1], s1[2]])
        # f1E = np.array(offsets["f1E"])
        # f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        # nE = np.array(offsets["nE"])
        # r = offsets["r"]
        # mighty_zap_len = offsets["mighty_zap_len"]

        # TODO: Replace the hard code
        m = [
            np.array([-0.0135, 0.018, 0.0555]),
            np.array([-0.0135, -0.018, 0.0355]),
        ]
        fE = [
            np.array([-0.01916, 0.018, -0.01567]),
            np.array([-0.01916, -0.018, -0.01567]),
        ]
        link_len = [0.059, 0.0395]
        nE = np.array([1, 0, 0])
        a = 0.02
        r = 0.01

        motor_pos_init: List[float] = []
        for joint_name in self.get_joint_attrs("is_closed_loop", True):
            if joint_name in ankle_pos:
                motor_pos_init.append(self.config[joint_name]["init_pos"])

        # Extract ankle pitch and roll from the input
        ankle_pitch, ankle_roll = ankle_pos.values()

        # Precompute cosine and sine for roll and pitch to use in rotation matrices
        cos_roll, sin_roll = np.cos(ankle_roll), np.sin(ankle_roll)
        cos_pitch, sin_pitch = np.cos(ankle_pitch), np.sin(ankle_pitch)

        # Roll rotation matrix
        R_roll = np.array(
            [[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]]
        )

        # Pitch rotation matrix
        R_pitch = np.array(
            [[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]]
        )

        # Combined rotation matrix
        R = R_roll @ R_pitch

        n_hat = R @ nE

        motor_pos: Dict[str, float] = {}
        for i, joint_name in enumerate(ankle_pos.keys()):
            f = R @ fE[i]
            delta = m[i] - f
            k = delta - np.dot(n_hat, delta) * n_hat
            d = delta - r * k / np.linalg.norm(k)
            c1 = -2 * a * d[0]
            c2 = 2 * a * d[2]
            c3 = np.sqrt(c1**2 + c2**2)
            c4 = a**2 + d[0] ** 2 + d[1] ** 2 + d[2] ** 2 - link_len[i] ** 2
            phi = np.arctan2(c2, c1)
            theta = phi + np.arccos(c4 / c3)  # cosine needs to be smaller than 0
            # TODO: Double check the computation here
            motor_pos_init_remainder = motor_pos_init[i] % (np.pi / 2)
            if motor_pos_init_remainder > np.pi / 4:
                motor_pos[joint_name] = theta % (np.pi / 2) - motor_pos_init_remainder
            else:
                motor_pos[joint_name] = (
                    np.pi / 2 - motor_pos_init_remainder - theta % (np.pi / 2)
                )

        return motor_pos

    def ankle_fk(self, motor_pos: List[float]) -> npt.NDArray[np.float32]:
        ankle_pos = self.ankle_fk_lookup_table(np.clip(motor_pos, 1, 4095))
        # Ensure the output is squeezed to a 1D array and handle NaN cases.
        ankle_pos = np.array(ankle_pos).squeeze()

        return ankle_pos

    # @profile()
    def precompute_ankle_fk_lookup(self, step_deg=0.5):
        step_rad = np.deg2rad(step_deg)
        pitch_limits = [
            self.config["joints"]["left_ank_pitch"]["lower_limit"],
            self.config["joints"]["left_ank_pitch"]["upper_limit"],
        ]
        roll_limits = [
            self.config["joints"]["left_ank_roll"]["lower_limit"],
            self.config["joints"]["left_ank_roll"]["upper_limit"],
        ]

        pitch_range = np.arange(pitch_limits[0], pitch_limits[1] + step_rad, step_rad)
        roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)
        pitch_grid, roll_grid = np.meshgrid(pitch_range, roll_range, indexing="ij")

        d1_values = np.zeros_like(pitch_grid)
        d2_values = np.zeros_like(roll_grid)
        for i in range(len(pitch_range)):
            for j in range(len(roll_range)):
                d1, d2 = self.ankle_ik([pitch_range[i], roll_range[j]])
                d1_values[i, j] = d1
                d2_values[i, j] = d2

        valid_mask = (
            (d1_values >= 0)
            & (d1_values <= 4096)
            & (d2_values >= 0)
            & (d2_values <= 4096)
        )

        # Filter out valid data points
        points = np.column_stack((d1_values[valid_mask], d2_values[valid_mask]))
        values = np.column_stack((pitch_grid[valid_mask], roll_grid[valid_mask]))

        return points, values

    def solve_ik(
        self,
        target_left_foot_pos: List[float],
        target_left_foot_ori: List[float],
        target_right_foot_pos: List[float],
        target_right_foot_ori: List[float],
        joint_angles_curr: List[float],
    ) -> List[float]:
        joint_angles = copy.deepcopy(joint_angles_curr)

        self._solve_leg_ik(
            target_left_foot_pos, target_left_foot_ori, "left", joint_angles
        )
        self._solve_leg_ik(
            target_right_foot_pos, target_right_foot_ori, "right", joint_angles
        )
        return joint_angles

    def _solve_leg_ik(
        self,
        target_foot_pos: Tuple[float, float, float],
        target_foot_ori: Tuple[float, float, float],
        side: str,
        joint_angles: Dict[str, float],
    ) -> List[float]:
        # Calculate leg angles
        angles_dict = self.config.compute_leg_angles(
            target_foot_pos, target_foot_ori, side, self.offsets
        )

        # Update joint angles based on calculations
        for name, angle in angles_dict.items():
            if f"{side}_{name}" in joint_angles:
                joint_angles[f"{side}_{name}"] = angle
            elif f"{side[0]}_{name}" in joint_angles:
                joint_angles[f"{side[0]}_{name}"] = angle
            else:
                raise ValueError(f"Joint '{name}' not found in joint angles.")

        return joint_angles
