import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator  # type: ignore
from yourdfpy import URDF  # type: ignore

from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import log


# TODO：actuator position to joint position
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

        self.root_path = os.path.join("toddlerbot", "robot_descriptions", self.name)
        self.config_file_path = os.path.join(self.root_path, "config.json")
        self.cache_file_path = os.path.join(self.root_path, f"{self.name}_data.pkl")

        self.load_robot_config()
        self.load_robot_data()

        self.init_joint_angles = self.initialize_joint_angles()
        motor_names = self.get_joint_attrs("is_passive", False)
        motor_ids = self.get_joint_attrs("is_passive", False, "id")
        self.motor_ordering = [name for _, name in sorted(zip(motor_ids, motor_names))]

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
                self.data_dict: Dict[str, Any] = pickle.load(f)
                log("Loaded cached data.", header="Robot")
        else:
            urdf_path = find_robot_file_path(self.name)
            urdf: URDF = URDF.load(urdf_path)  # type: ignore
            self.compute_data(urdf)

            with open(self.cache_file_path, "wb") as f:
                pickle.dump(self.data_dict, f)
                log("Computed and cached new data.", header="Robot")

        self.com = self.data_dict["com"]
        self.foot_size = self.data_dict["foot_size"]
        self.offsets = self.data_dict["offsets"]
        points, values = self.data_dict["ankle_fk_lookup_table"]
        self.ankle_fk_lookup_table = LinearNDInterpolator(points, values)

    def compute_data(self, urdf: URDF):
        self.data_dict: Dict[str, Any] = {}
        # TODO: check the value in MuJoCo
        self.data_dict["com"] = urdf.scene.center_mass
        self.data_dict["foot_size"] = self.compute_foot_size(urdf)
        self.data_dict["offsets"] = self.compute_offsets(urdf)
        self.data_dict["ank_act_zero"] = self.ankle_ik([0.0, 0.0])
        points, values = self.compute_ankle_fk_lookup()
        self.data_dict["ankle_fk_lookup_table"] = (points, values)

    def compute_foot_size(self, urdf: URDF) -> npt.NDArray[np.float32]:
        foot_bounds = urdf.scene.geometry.get("left_ank_pitch_link_visual.stl").bounds  # type: ignore
        foot_ori = urdf.scene.graph.get("ank_pitch_link")[0][:3, :3]  # type: ignore
        foot_bounds_rotated = foot_bounds @ foot_ori.T  # type: ignore
        foot_size = np.abs(foot_bounds_rotated[1] - foot_bounds_rotated[0])  # type: ignore

        # 0.004 is the thickness of the foot pad
        return np.array([foot_size[0], foot_size[1], 0.004])

    def compute_offsets(self, urdf: URDF):
        graph = urdf.scene.graph  # type: ignore

        offsets: Dict[str, Any] = {}

        ##### Below are for the leg IK #####
        # from the hip roll joint to the hip pitch joint
        offsets["hip_roll_to_pitch_z"] = (
            graph.get("hip_roll_link")[0][2, 3]  # type: ignore
            - graph.get("left_hip_pitch_link_xm430")[0][2, 3]  # type: ignore
        )
        # from the hip pitch joint to the knee joint
        offsets["hip_pitch_to_knee_z"] = (
            graph.get("left_hip_pitch_link_xm430")[0][2, 3]  # type: ignore
            - graph.get("left_calf_link")[0][2, 3]  # type: ignore
        )
        # from the knee joint to the ankle roll joint
        offsets["knee_to_ank_roll_z"] = (
            graph.get("left_calf_link")[0][2, 3] - graph.get("ank_pitch_link")[0][2, 3]  # type: ignore
        )
        # from the hip center to the foot
        offsets["y_offset_com_to_foot"] = graph.get("ank_pitch_link")[0][1, 3]  # type: ignore

        ##### Below are for the ankle IK #####
        # TODO: Remove hard-coded values
        ank_origin: npt.NDArray[np.float32] = np.array(
            graph.get("ank_pitch_link")[0][:3, 3]  # type: ignore
        )

        offsets["fE"] = [
            graph.get("ank_rr_link")[0][:3, 3] - ank_origin,  # type: ignore
            graph.get("ank_rr_link_2")[0][:3, 3] - ank_origin,  # type: ignore
        ]
        offsets["m"] = [
            graph.get("ank_motor_arm")[0][:3, 3] - ank_origin,  # type: ignore
            graph.get("ank_motor_arm_2")[0][:3, 3] - ank_origin,  # type: ignore
        ]
        offsets["m"][0][1] += 0.00582666
        offsets["m"][1][1] -= 0.00582666
        offsets["nE"] = np.array([1, 0, 0])
        offsets["link_len"] = [0.05900847, 0.03951266]
        offsets["a"] = 0.02
        offsets["r"] = 0.01

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
            if (
                key_name in joint_config
                and joint_config[key_name] == key_value
                and (joint_config["group"] == group or group == "all")
            ):
                if isinstance(attr_values, dict):
                    id = joint_config["id"]
                    self.config["joints"][joint_name][attr_name] = attr_values[id]
                else:
                    self.config["joints"][joint_name][attr_name] = attr_values[i]
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

    def ankle_ik(self, ankle_pos: List[float], side: str = "left") -> Dict[str, float]:
        # Extracting offset values and converting to NumPy arrays
        offsets = self.data_dict["offsets"]
        if "ank_act_zero" in self.data_dict:
            ank_act_zero = self.data_dict["ank_act_zero"]
        else:
            ank_act_zero = {"ank_act_1": 0, "ank_act_2": 0}

        # Extract ankle pitch and roll from the input
        ankle_pitch, ankle_roll = ankle_pos

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

        n_hat = R @ offsets["nE"]

        ank_act_dict: Dict[str, float] = {}
        for i, ank_act_name in enumerate(ank_act_zero.keys()):
            f = R @ offsets["fE"][i]
            delta = offsets["m"][i] - f
            k = delta - np.dot(n_hat, delta) * n_hat
            d = delta - offsets["r"] * k / np.linalg.norm(k)
            c1 = -2 * offsets["a"] * d[0]
            c2 = 2 * offsets["a"] * d[2]
            c3 = np.sqrt(c1**2 + c2**2)
            c4 = (
                offsets["a"] ** 2
                + d[0] ** 2
                + d[1] ** 2
                + d[2] ** 2
                - offsets["link_len"][i] ** 2
            )
            phi = np.arctan2(c2, c1)
            if -1 <= c4 / c3 <= 1:
                theta = np.pi - (phi + np.arccos(c4 / c3))
            else:
                theta = np.nan

            if (i == 0 and side == "left") or (i == 1 and side == "right"):
                ank_act_dict[ank_act_name] = -theta - ank_act_zero[ank_act_name]
            else:
                ank_act_dict[ank_act_name] = theta - ank_act_zero[ank_act_name]

        return ank_act_dict

    def ankle_fk(
        self, motor_pos: List[float], side: str = "left"
    ) -> npt.NDArray[np.float32]:
        if side == "right":
            motor_pos *= -1

        ankle_pos = self.ankle_fk_lookup_table(np.clip(motor_pos, 1, 4095))
        # Ensure the output is squeezed to a 1D array and handle NaN cases.
        ankle_pos = np.array(ankle_pos).squeeze()

        return ankle_pos

    # @profile()
    def compute_ankle_fk_lookup(self, step_degree: float = 0.5):
        step_rad = np.deg2rad(step_degree)
        pitch_limits = [
            self.config["joints"]["left_ank_pitch"]["lower_limit"],
            self.config["joints"]["left_ank_pitch"]["upper_limit"],
        ]
        roll_limits = [
            self.config["joints"]["left_ank_roll"]["lower_limit"],
            self.config["joints"]["left_ank_roll"]["upper_limit"],
        ]
        act_1_limits = [
            self.config["joints"]["left_ank_act_1"]["lower_limit"],
            self.config["joints"]["left_ank_act_1"]["upper_limit"],
        ]
        act_2_limits = [
            self.config["joints"]["left_ank_act_2"]["lower_limit"],
            self.config["joints"]["left_ank_act_2"]["upper_limit"],
        ]

        pitch_range = np.arange(pitch_limits[0], pitch_limits[1] + step_rad, step_rad)  # type: ignore
        roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)  # type: ignore
        pitch_grid, roll_grid = np.meshgrid(pitch_range, roll_range, indexing="ij")  # type: ignore

        act_1_values = np.zeros_like(pitch_grid)
        act_2_values = np.zeros_like(roll_grid)
        for i in range(len(pitch_range)):  # type: ignore
            for j in range(len(roll_range)):  # type: ignore
                ank_act_dict = self.ankle_ik([pitch_range[i], roll_range[j]])
                act_1_values[i, j] = ank_act_dict["ank_act_1"]
                act_2_values[i, j] = ank_act_dict["ank_act_2"]

        valid_mask = (
            (act_1_values >= act_1_limits[0])
            & (act_1_values <= act_1_limits[1])
            & (act_2_values >= act_2_limits[0])
            & (act_2_values <= act_2_limits[1])
        )

        # Filter out valid data points
        points = np.column_stack((act_1_values[valid_mask], act_2_values[valid_mask]))
        values = np.column_stack((pitch_grid[valid_mask], roll_grid[valid_mask]))

        return points, values

    # def solve_ik(
    #     self,
    #     target_left_foot_pos: List[float],
    #     target_left_foot_ori: List[float],
    #     target_right_foot_pos: List[float],
    #     target_right_foot_ori: List[float],
    #     joint_angles_curr: List[float],
    # ) -> List[float]:
    #     joint_angles = copy.deepcopy(joint_angles_curr)

    #     self._solve_leg_ik(
    #         target_left_foot_pos, target_left_foot_ori, "left", joint_angles
    #     )
    #     self._solve_leg_ik(
    #         target_right_foot_pos, target_right_foot_ori, "right", joint_angles
    #     )
    #     return joint_angles

    # def _solve_leg_ik(
    #     self,
    #     target_foot_pos: Tuple[float, float, float],
    #     target_foot_ori: Tuple[float, float, float],
    #     side: str,
    #     joint_angles: Dict[str, float],
    # ) -> List[float]:
    #     # Calculate leg angles
    #     angles_dict = self.config.compute_leg_angles(
    #         target_foot_pos, target_foot_ori, side, self.offsets
    #     )

    #     # Update joint angles based on calculations
    #     for name, angle in angles_dict.items():
    #         if f"{side}_{name}" in joint_angles:
    #             joint_angles[f"{side}_{name}"] = angle
    #         elif f"{side[0]}_{name}" in joint_angles:
    #             joint_angles[f"{side[0]}_{name}"] = angle
    #         else:
    #             raise ValueError(f"Joint '{name}' not found in joint angles.")

    #     return joint_angles
