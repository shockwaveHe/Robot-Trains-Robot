import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator  # type: ignore
from scipy.spatial import Delaunay  # type: ignore
from yourdfpy import URDF  # type: ignore

from toddlerbot.actuation import JointState
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import log  # , profile


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
        self.config_path = os.path.join(self.root_path, "config.json")
        self.collision_config_path = os.path.join(
            self.root_path, "config_collision.json"
        )
        self.cache_path = os.path.join(self.root_path, f"{self.name}_cache.pkl")

        self.load_robot_config()
        self.load_robot_cache()

    def load_robot_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")

        if os.path.exists(self.collision_config_path):
            with open(self.collision_config_path, "r") as f:
                self.collision_config = json.load(f)

        else:
            raise FileNotFoundError(
                f"No collision config file found for robot '{self.name}'."
            )

    def load_robot_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.data_dict: Dict[str, Any] = pickle.load(f)
                log("Loaded cached data.", header="Robot")
        else:
            urdf_path = find_robot_file_path(self.name)
            urdf: URDF = URDF.load(urdf_path)  # type: ignore
            self.compute_cache(urdf)

            with open(self.cache_path, "wb") as f:
                pickle.dump(self.data_dict, f)
                log("Computed and cached new data.", header="Robot")

        if "ank_fk_lookup_table" in self.data_dict:
            points, values = self.data_dict["ank_fk_lookup_table"]
            # TODO: Look up the speed
            self.ank_fk_lookup_table = LinearNDInterpolator(points, values)
            self.ank_act_pos_tri = Delaunay(points)
            self.ank_pos_tri = Delaunay(values)

    def compute_cache(self, urdf: URDF):
        self.data_dict: Dict[str, Any] = {"name": self.name}
        if "foot_name" in self.config["general"]:
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
        offsets["knee_to_ank_pitch_z"] = (
            graph.get("left_calf_link")[0][2, 3] - graph.get("ank_pitch_link")[0][2, 3]  # type: ignore
        )
        # from the hip center to the foot
        offsets["foot_to_com_x"] = graph.get("ank_pitch_link")[0][0, 3]  # type: ignore
        offsets["foot_to_com_y"] = graph.get("ank_pitch_link")[0][1, 3]  # type: ignore

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
                motor_angles[joint_name] = 0.0

        return motor_angles

    @property
    def default_motor_angles(self) -> Dict[str, float]:
        motor_angles: Dict[str, float] = {}
        for joint_name, joint_config in self.config["joints"].items():
            if not joint_config["is_passive"]:
                motor_angles[joint_name] = joint_config["default_pos"]

        return motor_angles

    @property
    def init_joint_angles(self) -> Dict[str, float]:
        return self.motor_to_joint_angles(self.init_motor_angles)

    @property
    def default_joint_angles(self) -> Dict[str, float]:
        return self.motor_to_joint_angles(self.default_motor_angles)

    @property
    def motor_ordering(self) -> List[str]:
        return list(self.init_motor_angles.keys())

    @property
    def joint_ordering(self) -> List[str]:
        return list(self.init_joint_angles.keys())

    @property
    def foot_name(self) -> str:
        return self.config["general"]["foot_name"]

    @property
    def foot_z(self) -> float:
        return self.config["general"]["offsets"]["foot_z"]

    @property
    def collider_names(self) -> List[str]:
        collider_names: List[str] = []
        for link_name, link_config in self.collision_config.items():
            if link_config["has_collision"]:
                collider_names.append(link_name)

        return collider_names

    @property
    def action_size(self) -> int:
        return len(self.motor_ordering)

    @property
    def joint_group(self) -> Dict[str, str]:
        joint_group: Dict[str, str] = {}
        for joint_name, joint_config in self.config["joints"].items():
            joint_group[joint_name] = joint_config["group"]

        return joint_group

    @property
    def joint_limits(self) -> Dict[str, List[float]]:
        joint_limits: Dict[str, List[float]] = {}
        for joint_name, joint_config in self.config["joints"].items():
            joint_limits[joint_name] = [
                joint_config["lower_limit"],
                joint_config["upper_limit"],
            ]

        return joint_limits

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
        waist_roll = offsets["waist_roll_coef"] * (-motor_pos[0] + motor_pos[1])
        waist_yaw = offsets["waist_yaw_coef"] * (-motor_pos[0] - motor_pos[1])
        return [waist_roll, waist_yaw]

    def waist_ik(self, waist_pos: List[float]) -> List[float]:
        offsets = self.config["general"]["offsets"]
        roll = waist_pos[0] / offsets["waist_roll_coef"]
        yaw = waist_pos[1] / offsets["waist_yaw_coef"]
        waist_act_1 = (-roll - yaw) / 2
        waist_act_2 = (roll - yaw) / 2
        return [waist_act_1, waist_act_2]

    def is_valid_waist_point(
        self, point: List[float], direction: str = "forward"
    ) -> bool:
        # Create Delaunay triangulation
        if direction == "forward":
            waist_roll, waist_yaw = self.waist_ik(point)
        else:
            waist_roll, waist_yaw = point

        roll_limits = self.joint_limits["waist_roll"]
        yaw_limits = self.joint_limits["waist_yaw"]
        if (
            waist_roll < roll_limits[0]
            or waist_roll > roll_limits[1]
            or waist_yaw < yaw_limits[0]
            or waist_yaw > yaw_limits[1]
        ):
            return False
        else:
            return True

    def sample_waist_point(self, direction: str = "forward") -> List[float]:
        if direction == "forward":
            min_bounds, max_bounds = zip(
                self.joint_limits["waist_act_1"], self.joint_limits["waist_act_2"]
            )
        else:
            min_bounds, max_bounds = zip(
                self.joint_limits["waist_roll"], self.joint_limits["waist_yaw"]
            )

        while True:
            # Generate a random point within the bounding box of the convex hull
            random_point = list(np.random.uniform(min_bounds, max_bounds))

            # Check if the point is within the convex hull
            if self.is_valid_waist_point(random_point, direction):
                return random_point

    def ankle_ik(self, ankle_pos: List[float], side: str = "left") -> List[float]:
        if side == "right":
            ankle_pos = [-ankle_pos[0], ankle_pos[1]]

        # Extracting offset values and converting to NumPy arrays
        offsets = self.data_dict["offsets"]
        if "ank_act_zero" in self.data_dict:
            ank_act_zero = self.data_dict["ank_act_zero"]
        else:
            ank_act_zero = [0, 0]

        # Extract ankle pitch and roll from the input
        ankle_roll, ankle_pitch = ankle_pos

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
        R_ankle = R_pitch @ R_roll

        n_hat = R_ankle @ offsets["nE"]

        ank_act_pos: List[float] = []
        for i in range(len(ank_act_zero)):
            f = R_ankle @ offsets["fE"][i]
            delta = offsets["m"][i] - f
            k = delta - np.dot(n_hat, delta) * n_hat  # type: ignore
            d = delta - offsets["r"] * k / np.linalg.norm(k)  # type: ignore
            a = 2 * offsets["a"] * d[0]
            b = 2 * offsets["a"] * d[2]
            R = np.sqrt(a**2 + b**2)
            c = (
                offsets["a"] ** 2
                + d[0] ** 2
                + d[1] ** 2
                + d[2] ** 2
                - offsets["rod_len"][i] ** 2
            )
            phi = np.arctan2(b, a)
            if c / R < -1 or c / R > 1:
                pos: float = np.nan
            else:
                theta_1 = phi + np.arccos(c / R)
                theta_2 = phi - np.arccos(c / R)

                theta = theta_1 if np.cos(theta_1) > 0 else theta_2
                if i == 0:
                    pos = -theta - ank_act_zero[i]
                else:
                    pos = theta - ank_act_zero[i]

                joint_limits = self.joint_limits[f"{side}_ank_act_{i+1}"]
                if pos < joint_limits[0] or pos > joint_limits[1]:
                    pos = np.nan

            ank_act_pos.append(pos)

        if side == "right":
            ank_act_pos = [-ank_act_pos[0], -ank_act_pos[1]]

        return ank_act_pos

    # @profile()
    def ankle_fk(self, motor_pos: List[float], side: str = "left") -> List[float]:
        if side == "right":
            motor_pos = [-motor_pos[0], -motor_pos[1]]

        ankle_pos_arr = self.ank_fk_lookup_table(motor_pos).squeeze()
        ankle_pos = ankle_pos_arr.tolist()

        if side == "right":
            ankle_pos = [-ankle_pos[0], ankle_pos[1]]

        return ankle_pos

    # @profile()
    def compute_ankle_fk_lookup(self, step_degree: float = 0.5):
        step_rad = np.deg2rad(step_degree)
        roll_limits = self.joint_limits["left_ank_roll"]
        pitch_limits = self.joint_limits["left_ank_pitch"]
        act_1_limits = self.joint_limits["left_ank_act_1"]
        act_2_limits = self.joint_limits["left_ank_act_2"]

        roll_range = np.arange(roll_limits[0], roll_limits[1] + step_rad, step_rad)  # type: ignore
        pitch_range = np.arange(pitch_limits[0], pitch_limits[1] + step_rad, step_rad)  # type: ignore
        roll_grid, pitch_grid = np.meshgrid(roll_range, pitch_range, indexing="ij")  # type: ignore

        act_1_grid = np.zeros_like(roll_grid)
        act_2_grid = np.zeros_like(pitch_grid)
        for i in range(len(roll_range)):  # type: ignore
            for j in range(len(pitch_range)):  # type: ignore
                act_pos: List[float] = self.ankle_ik([roll_range[i], pitch_range[j]])
                act_1_grid[i, j] = act_pos[0]
                act_2_grid[i, j] = act_pos[1]

        valid_mask = (
            (act_1_grid >= act_1_limits[0])
            & (act_1_grid <= act_1_limits[1])
            & (act_2_grid >= act_2_limits[0])
            & (act_2_grid <= act_2_limits[1])
        )

        # Filter out valid data points
        points = np.column_stack((act_1_grid[valid_mask], act_2_grid[valid_mask]))  # type: ignore
        values = np.column_stack((roll_grid[valid_mask], pitch_grid[valid_mask]))  # type: ignore

        return points, values

    def is_valid_ankle_point(
        self, point: List[float], direction: str = "forward", side: str = "left"
    ) -> bool:
        # Create Delaunay triangulation
        if direction == "forward":
            tri = self.ank_act_pos_tri
        else:
            tri = self.ank_pos_tri

        if tri.find_simplex(point) >= 0:  # type: ignore
            if direction == "forward":
                ankle_pos = self.ankle_fk(point, side)
                if np.isnan(ankle_pos).any():
                    return False
                else:
                    ank_act_pos = self.ankle_ik(ankle_pos, side)
                    if np.allclose(ank_act_pos, point, atol=1e-3):  # type: ignore
                        return True
                    else:
                        return False
            else:
                if np.isnan(self.ankle_ik(point, side)).any():
                    return False
                else:
                    return True
        else:
            return False

    def sample_ankle_point(
        self, direction: str = "forward", side: str = "left"
    ) -> List[float]:
        if direction == "forward":
            min_bounds, max_bounds = zip(
                self.joint_limits[f"{side}_ank_act_1"],
                self.joint_limits[f"{side}_ank_act_2"],
            )
        else:
            min_bounds, max_bounds = zip(
                self.joint_limits[f"{side}_ank_roll"],
                self.joint_limits[f"{side}_ank_pitch"],
            )

        while True:
            # Generate a random point within the bounding box of the convex hull
            random_point = list(np.random.uniform(min_bounds, max_bounds))

            # Check if the point is within the convex hull
            if self.is_valid_ankle_point(random_point, direction, side):
                return random_point

    # @profile()
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
                    joint_angles["left_ank_roll"] = 0.0
                    joint_angles["left_ank_pitch"] = 0.0
                    left_ank_act_pos.append(motor_pos)
                elif "right" in motor_name:
                    joint_angles["right_ank_roll"] = 0.0
                    joint_angles["right_ank_pitch"] = 0.0
                    right_ank_act_pos.append(motor_pos)
            elif transmission == "none":
                joint_angles[motor_name] = motor_pos

        if len(waist_act_pos) > 0:
            joint_angles["waist_roll"], joint_angles["waist_yaw"] = self.waist_fk(
                waist_act_pos
            )

        if len(left_ank_act_pos) > 0:
            joint_angles["left_ank_roll"], joint_angles["left_ank_pitch"] = (
                self.ankle_fk(left_ank_act_pos, "left")
            )

        if len(right_ank_act_pos) > 0:
            joint_angles["right_ank_roll"], joint_angles["right_ank_pitch"] = (
                self.ankle_fk(right_ank_act_pos, "right")
            )

        return joint_angles

    # @profile()
    def motor_to_joint_state(
        self,
        motor_state_dict: Dict[str, JointState],
        last_joint_state_dict: Dict[str, JointState],
    ) -> Dict[str, JointState]:
        joint_state_dict: Dict[str, JointState] = {}
        joints_config = self.config["joints"]
        waist_act_pos: List[float] = []
        left_ank_act_pos: List[float] = []
        right_ank_act_pos: List[float] = []
        for motor_name, motor_state in motor_state_dict.items():
            transmission = joints_config[motor_name]["transmission"]
            if transmission == "gears":
                joint_name = motor_name.replace("_drive", "_driven")
                joint_state_dict[joint_name] = JointState(
                    time=motor_state.time,
                    pos=motor_state.pos * joints_config[motor_name]["gear_ratio"],
                    vel=-motor_state.vel / joints_config[motor_name]["gear_ratio"],
                )
            elif transmission == "waist":
                # Placeholder to ensure the correct order
                joint_state_dict["waist_roll"] = JointState(
                    time=motor_state.time, pos=0.0, vel=0.0
                )
                joint_state_dict["waist_yaw"] = JointState(
                    time=motor_state.time, pos=0.0, vel=0.0
                )
                waist_act_pos.append(motor_state.pos)
            elif transmission == "knee":
                joint_name: str = motor_name.replace("_act", "_pitch")
                joint_state_dict[joint_name] = JointState(
                    time=motor_state.time, pos=motor_state.pos, vel=motor_state.vel
                )
            elif transmission == "ankle":
                if "left" in motor_name:
                    joint_state_dict["left_ank_roll"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    joint_state_dict["left_ank_pitch"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    left_ank_act_pos.append(motor_state.pos)
                elif "right" in motor_name:
                    joint_state_dict["right_ank_roll"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    joint_state_dict["right_ank_pitch"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    right_ank_act_pos.append(motor_state.pos)
            elif transmission == "none":
                joint_state_dict[motor_name] = JointState(
                    time=motor_state.time, pos=motor_state.pos, vel=motor_state.vel
                )

        if len(waist_act_pos) > 0:
            joint_state_dict["waist_roll"].pos, joint_state_dict["waist_yaw"].pos = (
                self.waist_fk(waist_act_pos)
            )
        if len(left_ank_act_pos) > 0:
            (
                joint_state_dict["left_ank_roll"].pos,
                joint_state_dict["left_ank_pitch"].pos,
            ) = self.ankle_fk(left_ank_act_pos, "left")
        if len(right_ank_act_pos) > 0:
            (
                joint_state_dict["right_ank_roll"].pos,
                joint_state_dict["right_ank_pitch"].pos,
            ) = self.ankle_fk(right_ank_act_pos, "right")

        for joint_name in [
            "waist_roll",
            "waist_yaw",
            "left_ank_roll",
            "left_ank_pitch",
            "right_ank_roll",
            "right_ank_pitch",
        ]:
            if last_joint_state_dict and joint_name in last_joint_state_dict:
                time_delta = (
                    joint_state_dict[joint_name].time
                    - last_joint_state_dict[joint_name].time
                )
                if time_delta > 0:
                    joint_state_dict[joint_name].vel = (
                        joint_state_dict[joint_name].pos
                        - last_joint_state_dict[joint_name].pos
                    ) / time_delta
                else:
                    raise ValueError("Time delta must be greater than 0.")

        return joint_state_dict

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

        motor_angles["waist_act_1"], motor_angles["waist_act_2"] = self.waist_ik(
            waist_pos
        )
        motor_angles["left_ank_act_1"], motor_angles["left_ank_act_2"] = self.ankle_ik(
            left_ankle_pos, "left"
        )
        motor_angles["right_ank_act_1"], motor_angles["right_ank_act_2"] = (
            self.ankle_ik(right_ankle_pos, "right")
        )

        return motor_angles

    def sample_motor_angles(self) -> Dict[str, float]:
        random_motor_angles: Dict[str, float] = {}
        for motor_name in self.motor_ordering:
            random_motor_angles[motor_name] = np.random.uniform(
                self.joint_limits[motor_name][0],
                self.joint_limits[motor_name][1],
            )

        random_motor_angles["left_ank_act_1"], random_motor_angles["left_ank_act_2"] = (
            self.sample_ankle_point(side="left")
        )
        (
            random_motor_angles["right_ank_act_1"],
            random_motor_angles["right_ank_act_2"],
        ) = self.sample_ankle_point(side="right")

        random_motor_angles["waist_act_1"], random_motor_angles["waist_act_2"] = (
            self.sample_waist_point()
        )

        return random_motor_angles
