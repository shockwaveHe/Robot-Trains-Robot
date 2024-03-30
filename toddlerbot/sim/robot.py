import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from yourdfpy import URDF

from toddlerbot.robot_descriptions.robot_configs import robot_configs
from toddlerbot.utils.file_utils import find_description_path


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
        offsets["mighty_zap_len"] = 0.07521
        # np.linalg.norm(
        #     graph.get("ball_joint_ball")[0][:3, 3]
        #     - graph.get("12lf_rod_end")[0][:3, 3]
        # )
        # - 0.01369  # This number is read from onshape

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
                "lowerLimit": joint.limit.lower,
                "upperLimit": joint.limit.upper,
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

        return joint_angles

    def solve_ik(
        self,
        target_left_foot_pos: List[float],
        target_left_foot_ori: List[float],
        target_right_foot_pos: List[float],
        target_right_foot_ori: List[float],
        joint_angles_curr: List[float],
    ) -> List[float]:
        """
        Solves inverse kinematics for the given foot positions and orientations.

        Args:
            target_left_foot_pos (List[float]): Target position for the left foot (x, y, z).
            target_left_foot_ori (List[float]): Target orientation for the left foot (quaternion).
            target_right_foot_pos (List[float]): Target position for the right foot (x, y, z).
            target_right_foot_ori (List[float]): Target orientation for the right foot (quaternion).
            current_angles (List[float], optional): Current joint angles. Defaults to None.

        Returns:
            List[float]: New joint angles to achieve the desired foot positions and orientations.

        Raises:
            ValueError: If the robot has not been loaded yet.
        """
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
        """
        Calculates the leg angles based on the target foot position and orientation.

        Args:
            target_foot_pos (Tuple[float, float, float]): Target position for the foot (x, y, z).
            target_foot_ori (Tuple[float, float, float]): Target orientation for the foot (roll, pitch, yaw).
            side (str): The side of the leg ('left' or 'right').
            joint_angles (List[float]): Current list of joint angles.

        Returns:
            List[float]: Updated list of joint angles after calculation.
        """
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
    import numpy as np

    from toddlerbot.sim.pybullet_sim import PyBulletSim
    from toddlerbot.utils.math_utils import round_floats

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
