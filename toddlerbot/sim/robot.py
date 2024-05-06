import copy
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from yourdfpy import URDF

from toddlerbot.robot_descriptions.robot_configs import robot_configs
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import log


class HumanoidRobot:
    """Class representing a humanoid robot."""

    def __init__(self, robot_name: str):
        """
        Initialize a humanoid robot with the given name.

        Args:
            robot_name (str): The name of the robot.

        Raises:
            ValueError: If the robot name is not supported.
        """
        if robot_name not in robot_configs:
            raise ValueError(f"Robot '{robot_name}' is not supported.")

        self.id = 0
        self.name = robot_name
        self.config = robot_configs[robot_name]

        self.load_robot_data()

        self.dynamixel_joint2id = {}
        self.sunny_sky_joint2id = {}
        self.mighty_zap_joint2id = {}
        for name, motor_param in self.config.motor_params.items():
            if motor_param.brand == "dynamixel":
                self.dynamixel_joint2id[name] = motor_param.id
            elif motor_param.brand == "sunny_sky":
                self.sunny_sky_joint2id[name] = motor_param.id
            elif motor_param.brand == "mighty_zap":
                self.mighty_zap_joint2id[name] = motor_param.id
            else:
                raise ValueError(f"Motor brand '{motor_param.brand}' is not supported.")

        self.dynamixel_id2joint = {v: k for k, v in self.dynamixel_joint2id.items()}
        self.sunny_sky_id2joint = {v: k for k, v in self.sunny_sky_joint2id.items()}
        self.mighty_zap_id2joint = {v: k for k, v in self.mighty_zap_joint2id.items()}

        self.ankle2mighty_zap = {"left": [0, 1], "right": [2, 3]}

    def load_robot_data(self):
        cache_file_path = os.path.join(
            "toddlerbot", "robot_descriptions", self.name, f"{self.name}_data.pkl"
        )

        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as f:
                robot_data = pickle.load(f)
                log("Loaded cached data.", header="Robot")

            self.joints_info = robot_data["joints_info"]
            self.com = robot_data["com"]
            self.foot_size = robot_data["foot_size"]
            self.offsets = robot_data["offsets"]
            points, values = robot_data["ankle_fk_lookup_table"]

        else:
            urdf_path = find_robot_file_path(self.name)
            urdf = URDF.load(urdf_path)
            self.joints_info = self.get_joint_info(urdf)

            if self.config.com is None:
                self.com = urdf.scene.center_mass
            else:
                self.com = self.config.com

            if self.config.foot_size is None:
                self.foot_size = self.compute_foot_size(urdf)
            else:
                self.foot_size = self.config.foot_size

            if self.config.offsets is None:
                self.offsets = self.compute_offsets(urdf)
            else:
                self.offsets = self.config.offsets

            points, values = self.precompute_ankle_fk_lookup()

            with open(cache_file_path, "wb") as f:
                robot_data = {
                    "joints_info": self.joints_info,
                    "com": self.com,
                    "foot_size": self.foot_size,
                    "offsets": self.offsets,
                    "ankle_fk_lookup_table": (points, values),
                }
                pickle.dump(robot_data, f)

                log("Computed and cached new data.", header="Robot")

        self.ankle_fk_lookup_table = LinearNDInterpolator(points, values)

    def compute_offsets(self, urdf):
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

    def compute_foot_size(self, urdf):
        foot_bounds = urdf.scene.geometry.get("left_ank_roll_link_visual.stl").bounds
        foot_ori = urdf.scene.graph.get("ank_roll_link")[0][:3, :3]
        foot_bounds_rotated = foot_bounds @ foot_ori.T
        foot_size = np.abs(foot_bounds_rotated[1] - foot_bounds_rotated[0])

        # 0.004 is the thickness of the foot pad
        return np.array([foot_size[0], foot_size[1], 0.004])

    def get_joint_info(self, urdf):
        joint_info_dict = {}
        for joint, angle in zip(urdf.actuated_joints, urdf.cfg):
            joint_info_dict[joint.name] = {
                "init_angle": angle,
                "type": joint.type,
                "lower_limit": joint.limit.lower,
                "upper_limit": joint.limit.upper,
                "active": joint.name in self.config.motor_params.keys(),
            }

        sorted_joint_info_dict = {}
        for joint_name in self.config.motor_params:
            sorted_joint_info_dict[joint_name] = joint_info_dict[joint_name]

        return sorted_joint_info_dict

    # @profile()
    def precompute_ankle_fk_lookup(self, step_deg=0.5):
        step_rad = np.deg2rad(step_deg)
        pitch_limits = [
            self.joints_info["left_ank_pitch"]["lower_limit"],
            self.joints_info["left_ank_pitch"]["upper_limit"],
        ]
        roll_limits = [
            self.joints_info["left_ank_roll"]["lower_limit"],
            self.joints_info["left_ank_roll"]["upper_limit"],
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

    def ankle_fk(self, mighty_zap_pos):
        ankle_pos = self.ankle_fk_lookup_table(mighty_zap_pos)
        # Ensure the output is squeezed to a 1D array and handle NaN cases.
        ankle_pos = np.array(ankle_pos).squeeze()

        # Check if any part of ankle_pos is NaN, and replace it with 0.
        if np.any(np.isnan(ankle_pos)):
            # You can log or print a warning here if needed
            log(
                "NaN encountered in ankle_fk calculations, returning 0.",
                header="Robot",
                level="warning",
            )
            return [0] * len(ankle_pos)

        return list(ankle_pos)

    def ankle_ik(self, ankle_pos):
        # Extracting offset values and converting to NumPy arrays
        offsets = self.offsets
        s1 = np.array(offsets["s1"])
        s2 = np.array([s1[0], -s1[1], s1[2]])
        f1E = np.array(offsets["f1E"])
        f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        nE = np.array(offsets["nE"])
        r = offsets["r"]
        mighty_zap_len = offsets["mighty_zap_len"]

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

        # Rotated vectors
        n_hat = R @ nE
        f1 = R @ f1E
        f2 = R @ f2E

        # Delta calculations
        delta1 = s1 - f1
        delta2 = s2 - f2

        # Distance calculations
        d1_raw = np.sqrt(
            np.dot(n_hat, delta1) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta1)) - r) ** 2
        )
        d2_raw = np.sqrt(
            np.dot(n_hat, delta2) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta2)) - r) ** 2
        )

        # Final distance adjustments
        scale_factor = 1.365 * 1e5
        d1 = (d1_raw - mighty_zap_len) * scale_factor
        d2 = (d2_raw - mighty_zap_len) * scale_factor

        return [d1, d2]

    def initialize_joint_angles(self):
        joint_angles = {}
        for name, info in self.joints_info.items():
            if info["active"]:
                joint_angles[name] = info["init_angle"]

        initial_joint_angles = copy.deepcopy(joint_angles)
        if self.name == "robotis_op3":
            initial_joint_angles["l_sho_roll"] = np.pi / 2
            initial_joint_angles["r_sho_roll"] = -np.pi / 2
        elif self.name == "toddlerbot":
            initial_joint_angles["left_sho_roll"] = -np.pi / 2
            initial_joint_angles["right_sho_roll"] = np.pi / 2

        return joint_angles, initial_joint_angles

    def solve_ik(
        self,
        target_left_foot_pos: List[float],
        target_left_foot_ori: List[float],
        target_right_foot_pos: List[float],
        target_right_foot_ori: List[float],
        joint_angles_curr: List[float],
    ) -> List[float]:
        if self.id is None or self.joints_info is None:
            raise ValueError("Robot has not been loaded yet.")

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
