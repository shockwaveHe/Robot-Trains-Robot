import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import root
from yourdfpy import URDF

from toddlerbot.robot_descriptions.robot_configs import robot_configs
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log


@dataclass
class JointState:
    time: float
    pos: float


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

        self.name = robot_name
        self.config = robot_configs[robot_name]
        urdf_path = find_description_path(robot_name)
        self.urdf = URDF.load(urdf_path)

        self.id = 0
        self.joints_info = self.get_joint_info()
        if self.config.com is None:
            self.com = self.urdf.scene.center_mass
        else:
            self.com = self.config.com

        if self.config.foot_size is None:
            self.foot_size = self.compute_foot_size()
        else:
            self.foot_size = self.config.foot_size

        if self.config.offsets is None:
            self.offsets = self.compute_offsets()
        else:
            self.offsets = self.config.offsets

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

        self.ankle2mighty_zap = [[0, 1], [2, 3]]

    def compute_offsets(self):
        graph = self.urdf.scene.graph

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

        # Measured in the real robot (m)
        offsets["mighty_zap_len"] = 0.078

        return offsets

    def compute_foot_size(self):
        foot_bounds = self.urdf.scene.geometry.get(
            "left_ank_roll_link_visual.stl"
        ).bounds
        foot_ori = self.urdf.scene.graph.get("ank_roll_link")[0][:3, :3]
        foot_bounds_rotated = foot_bounds @ foot_ori.T
        foot_size = np.abs(foot_bounds_rotated[1] - foot_bounds_rotated[0])
        # 0.004 is the thickness of the foot pad
        return np.array([foot_size[0], foot_size[1], 0.004])

    def get_joint_info(self):
        joint_info_dict = {}
        for joint, angle in zip(self.urdf.actuated_joints, self.urdf.cfg):
            joint_info_dict[joint.name] = {
                "init_angle": angle,
                "type": joint.type,
                "lower_limit": joint.limit.lower,
                "upper_limit": joint.limit.upper,
                "active": joint.name in self.config.motor_params.keys(),
            }

        def get_brand(joint_name):
            if joint_name in self.config.motor_params:
                return self.config.motor_params[joint_name].brand
            return "default"

        sorted_joint_info = sorted(
            joint_info_dict.items(), key=lambda item: (get_brand(item[0]), item[0])
        )
        sorted_joint_info_dict = {
            joint_name: info for joint_name, info in sorted_joint_info
        }

        return sorted_joint_info_dict

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

    def ankle_fk(self, mighty_zap_pos, last_mighty_zap_pos):
        def objective_function(ankle_pos, target_pos):
            pos = self.ankle_ik(ankle_pos)
            error = np.array(pos) - np.array(target_pos)
            return error

        result = root(
            lambda x: objective_function(x, mighty_zap_pos),
            last_mighty_zap_pos,
            method="hybr",
            options={"xtol": 1e-6},
        )

        if result.success:
            optimized_ankle_pos = result.x
            return optimized_ankle_pos
        else:
            log("Solving ankle position failed", header="MightyZap", level="warning")
            return last_mighty_zap_pos

    def ankle_ik(self, ankle_pos):
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.

        offsets = self.offsets
        s1 = np.array(offsets["s1"])
        s2 = np.array([s1[0], -s1[1], s1[2]])
        f1E = np.array(offsets["f1E"])
        f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        nE = np.array(offsets["nE"])
        r = offsets["r"]
        mighty_zap_len = offsets["mighty_zap_len"]

        ankle_pitch = ankle_pos[0]
        ankle_roll = ankle_pos[1]
        R_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(ankle_roll), -np.sin(ankle_roll)],
                [0, np.sin(ankle_roll), np.cos(ankle_roll)],
            ]
        )
        R_pitch = np.array(
            [
                [np.cos(ankle_pitch), 0, np.sin(ankle_pitch)],
                [0, 1, 0],
                [-np.sin(ankle_pitch), 0, np.cos(ankle_pitch)],
            ]
        )
        R = np.dot(R_roll, R_pitch)
        n_hat = np.dot(R, nE)
        f1 = np.dot(R, f1E)
        f2 = np.dot(R, f2E)
        delta1 = s1 - f1
        delta2 = s2 - f2

        d1_raw = np.sqrt(
            np.dot(n_hat, delta1) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta1)) - r) ** 2
        )
        d2_raw = np.sqrt(
            np.dot(n_hat, delta2) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta2)) - r) ** 2
        )
        # 1.365 for 12Lf
        d1 = (d1_raw - mighty_zap_len) * 1.365 * 1e5
        d2 = (d2_raw - mighty_zap_len) * 1.365 * 1e5

        return [d1, d2]

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


# Example usage
if __name__ == "__main__":
    from toddlerbot.sim.pybullet_sim import PyBulletSim

    robot = HumanoidRobot("toddlerbot")

    robot.compute_offsets()

    sim = PyBulletSim(robot)

    # Define target positions and orientations for left and right feet
    target_left_foot_pos = [0.2, 0.1, -0.2]
    target_right_foot_pos = [0.2, -0.1, -0.2]

    joint_angles = robot.solve_ik(
        target_left_foot_pos,
        [0, 0, 0],
        target_right_foot_pos,
        [0, 0, 0],
    )

    rounded_joint_angles = round_floats(joint_angles, 3)
    print(f"Joint angles: {rounded_joint_angles}")
