import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator  # type: ignore
from yourdfpy import URDF  # type: ignore

from toddlerbot.actuation import JointState
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

        self.root_path = os.path.join("toddlerbot", "robot_descriptions", self.name)
        self.config_file_path = os.path.join(self.root_path, "config.json")
        self.cache_file_path = os.path.join(self.root_path, f"{self.name}_data.pkl")

        self.load_robot_config()
        self.load_robot_data()

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

        points, values = self.data_dict["ank_fk_lookup_table"]
        self.ank_fk_lookup_table = LinearNDInterpolator(points, values)

    def compute_data(self, urdf: URDF):
        self.data_dict: Dict[str, Any] = {}
        self.data_dict["foot_size"] = self.compute_foot_size(urdf)
        self.data_dict["offsets"] = self.compute_offsets(urdf)
        self.data_dict["ank_act_zero"] = self.ankle_ik([0.0, 0.0])
        points, values = self.compute_ankle_fk_lookup()
        self.data_dict["ank_fk_lookup_table"] = (points, values)

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
        ank_act_arm_y = self.config["general"]["offsets"]["ank_act_arm_y"]
        offsets["m"][0][1] += ank_act_arm_y
        offsets["m"][1][1] -= ank_act_arm_y
        offsets["nE"] = np.array([1, 0, 0])
        offsets["rod_len"] = [
            self.config["general"]["offsets"]["ank_long_rod_len"],
            self.config["general"]["offsets"]["ank_short_rod_len"],
        ]
        offsets["a"] = self.config["general"]["offsets"]["ank_act_arm_r"]
        offsets["r"] = self.config["general"]["offsets"]["ank_rev_r"]

        return offsets

    @property
    def init_motor_angles(self) -> Dict[str, float]:
        motor_angles: Dict[str, float] = {}
        for joint_name, joint_config in self.config["joints"].items():
            if not joint_config["is_passive"]:
                motor_angles[joint_name] = joint_config["default_pos"]

        return motor_angles

    @property
    def init_joint_angles(self) -> Dict[str, float]:
        return self.motor_to_joint_angles(self.init_motor_angles)

    @property
    def motor_ordering(self) -> List[str]:
        return list(self.init_motor_angles.keys())

    @property
    def joint_ordering(self) -> List[str]:
        return list(self.init_joint_angles.keys())

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

    def waist_fk(self, motor_pos: List[float]) -> List[float]:
        offsets = self.config["general"]["offsets"]
        waist_roll = offsets["waist_roll_coef"] * (motor_pos[0] + motor_pos[1])
        waist_yaw = offsets["waist_yaw_coef"] * (motor_pos[0] - motor_pos[1])
        return [waist_roll, waist_yaw]

    def waist_ik(self, waist_pos: List[float]) -> List[float]:
        offsets = self.config["general"]["offsets"]
        roll = waist_pos[0] / offsets["waist_roll_coef"]
        yaw = waist_pos[1] / offsets["waist_yaw_coef"]
        waist_act_1 = (roll + yaw) / 2
        waist_act_2 = (roll - yaw) / 2
        return [waist_act_1, waist_act_2]

    def ankle_ik(self, ankle_pos: List[float], side: str = "left") -> List[float]:
        # Extracting offset values and converting to NumPy arrays
        offsets = self.data_dict["offsets"]
        if "ank_act_zero" in self.data_dict:
            ank_act_zero = self.data_dict["ank_act_zero"]
        else:
            ank_act_zero = [0, 0]

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
        R = R_pitch @ R_roll

        n_hat = R @ offsets["nE"]

        ank_act_pos: List[float] = []
        for i in range(len(ank_act_zero)):
            f = R @ offsets["fE"][i]
            delta = offsets["m"][i] - f
            k = delta - np.dot(n_hat, delta) * n_hat
            d = delta - offsets["r"] * k / np.linalg.norm(k)
            d_sq = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
            d_norm = np.sqrt(d_sq)
            c1 = 2 * offsets["a"] * d_norm
            c2 = offsets["a"] ** 2 + d_sq - offsets["rod_len"][i] ** 2
            if c2 < -c1 or c2 > c1:
                theta = np.nan
            else:
                alpha = np.arccos(d[0] / d_norm)
                beta = np.arccos(c2 / c1)
                theta = beta - alpha

            if (i == 0 and side == "left") or (i == 1 and side == "right"):
                pos = theta - ank_act_zero[i]
            else:
                pos = -theta - ank_act_zero[i]

            lower_limit = self.config["joints"][f"{side}_ank_act_{i+1}"]["lower_limit"]
            upper_limit = self.config["joints"][f"{side}_ank_act_{i+1}"]["upper_limit"]
            if pos < lower_limit or pos > upper_limit:
                pos = np.nan

            ank_act_pos.append(pos)

        return ank_act_pos

    def ankle_fk(self, motor_pos: List[float], side: str = "left") -> List[float]:
        if side == "right":
            motor_pos = [-motor_pos[0], -motor_pos[1]]

        ankle_pos_arr = self.ank_fk_lookup_table(motor_pos).squeeze()
        ankle_pos = ankle_pos_arr.tolist()

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
                ank_act_pos = self.ankle_ik([pitch_range[i], roll_range[j]])
                act_1_values[i, j] = ank_act_pos[0]
                act_2_values[i, j] = ank_act_pos[1]

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

    def joint_state_to_obs_arr(
        self, joint_state_dict: Dict[str, JointState]
    ) -> Dict[str, npt.NDArray[np.float32]]:
        time_obs: List[float] = []
        q_obs: List[float] = []
        dq_obs: List[float] = []
        for joint_name in joint_state_dict:
            time_obs.append(joint_state_dict[joint_name].time)
            q_obs.append(joint_state_dict[joint_name].pos)
            dq_obs.append(joint_state_dict[joint_name].vel)

        return {
            "time": np.array(time_obs, dtype=np.float32),
            "q": np.array(q_obs, dtype=np.float32),
            "dq": np.array(dq_obs, dtype=np.float32),
        }

    def motor_to_joint_angles(self, motor_angles: Dict[str, float]) -> Dict[str, float]:
        joint_angles: Dict[str, float] = {}
        joints_config = self.config["joints"]
        waist_act_pos: List[float] = []
        left_ank_act_pos: List[float] = []
        right_ank_act_pos: List[float] = []
        for motor_name, motor_pos in motor_angles.items():
            transmission = joints_config[motor_name]["transmission"]
            if transmission == "gears":
                joint_name = motor_name.replace("_drive", "_driven")
                joint_angles[joint_name] = (
                    motor_pos * joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "waist":
                # Placeholder to ensure the correct order
                joint_angles["waist_roll"] = 0.0
                joint_angles["waist_yaw"] = 0.0
                waist_act_pos.append(motor_pos)
            elif transmission == "knee":
                joint_name: str = motor_name.replace("_act", "_pitch")
                joint_angles[joint_name] = motor_pos
            elif transmission == "ankle":
                if "left" in motor_name:
                    joint_angles["left_ank_pitch"] = 0.0
                    joint_angles["left_ank_roll"] = 0.0
                    left_ank_act_pos.append(motor_pos)
                elif "right" in motor_name:
                    joint_angles["right_ank_pitch"] = 0.0
                    joint_angles["right_ank_roll"] = 0.0
                    right_ank_act_pos.append(motor_pos)
            elif transmission == "none":
                joint_angles[motor_name] = motor_pos

        joint_angles["waist_roll"], joint_angles["waist_yaw"] = self.waist_fk(
            waist_act_pos
        )
        joint_angles["left_ank_pitch"], joint_angles["left_ank_roll"] = self.ankle_fk(
            left_ank_act_pos, "left"
        )
        joint_angles["right_ank_pitch"], joint_angles["right_ank_roll"] = self.ankle_fk(
            right_ank_act_pos, "right"
        )

        return joint_angles

    def joint_to_motor_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        motor_angles: Dict[str, float] = {}
        joints_config = self.config["joints"]
        waist_pos: List[float] = []
        left_ankle_pos: List[float] = []
        right_ankle_pos: List[float] = []
        for joint_name, joint_pos in joint_angles.items():
            transmission = joints_config[joint_name]["transmission"]
            if transmission == "gears":
                motor_name = joint_name.replace("_driven", "_drive")
                motor_angles[motor_name] = (
                    joint_pos / joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "waist":
                # Placeholder to ensure the correct order
                motor_angles["waist_act_1"] = 0.0
                motor_angles["waist_act_2"] = 0.0
                waist_pos.append(joint_pos)
            elif transmission == "knee":
                motor_name: str = joint_name.replace("_pitch", "_act")
                motor_angles[motor_name] = joint_pos
            elif transmission == "ankle":
                if "left" in joint_name:
                    motor_angles["left_ank_act_1"] = 0.0
                    motor_angles["left_ank_act_2"] = 0.0
                    left_ankle_pos.append(joint_pos)
                elif "right" in joint_name:
                    motor_angles["right_ank_act_1"] = 0.0
                    motor_angles["right_ank_act_2"] = 0.0
                    right_ankle_pos.append(joint_pos)
            elif transmission == "none":
                motor_angles[joint_name] = joint_pos

        joint_angles["waist_act_1"], joint_angles["waist_act_2"] = self.waist_ik(
            waist_pos
        )
        joint_angles["left_ank_act_1"], joint_angles["left_ank_act_2"] = self.ankle_ik(
            left_ankle_pos, "left"
        )
        joint_angles["right_ank_act_1"], joint_angles["right_ank_act_2"] = (
            self.ankle_ik(right_ankle_pos, "right")
        )

        return joint_angles
