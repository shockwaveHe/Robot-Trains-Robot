import json
import os
from typing import Any, Dict, List


class Robot:
    """This class defines some data strucutres, FK, IK of ToddlerBot."""

    def __init__(self, robot_name: str):
        """Initializes a robot with specified configurations and paths.

        Args:
            robot_name (str): The name of the robot, used to set up directory paths and configurations.
        """
        self.id = 0
        self.name = robot_name

        self.root_path = os.path.join("toddlerbot", "descriptions", self.name)
        self.config_path = os.path.join(self.root_path, "config.json")
        self.collision_config_path = os.path.join(
            self.root_path, "config_collision.json"
        )
        self.cache_path = os.path.join(self.root_path, f"{self.name}_cache.pkl")

        self.load_robot_config()

        self.initialize()

    def load_robot_config(self):
        """Load the robot's configuration and collision configuration from JSON files.

        Raises:
            FileNotFoundError: If the main configuration file or the collision configuration file does not exist at the specified paths.
        """
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

    def initialize(self) -> None:
        """Initializes the robot's joint and motor configurations based on the provided configuration data.

        This method sets up the initial and default motor angles, establishes mappings between motors and joints, identifies passive joints, and configures collision detection. It also determines the presence of a gripper and sets joint groups and limits.

        Attributes:
            init_motor_angles (Dict[str, float]): Initial motor angles for active joints.
            init_joint_angles (Dict[str, float]): Initial joint angles derived from motor angles.
            motor_ordering (List[str]): Order of motors based on initial configuration.
            joint_ordering (List[str]): Order of joints based on initial configuration.
            default_motor_angles (Dict[str, float]): Default motor angles for active joints.
            default_joint_angles (Dict[str, float]): Default joint angles derived from motor angles.
            motor_to_joint_name (Dict[str, List[str]]): Mapping from motor names to joint names.
            joint_to_motor_name (Dict[str, List[str]]): Mapping from joint names to motor names.
            passive_joint_names (List[str]): Names of passive joints.
            foot_name (str): Name of the foot, if specified in the configuration.
            has_gripper (bool): Indicates if the robot has a gripper.
            collider_names (List[str]): Names of links with collision enabled.
            nu (int): Number of active motors.
            joint_groups (Dict[str, str]): Group classification for each joint.
            joint_limits (Dict[str, List[float]]): Lower and upper limits for each joint.
        """
        self.init_motor_angles: Dict[str, float] = {}
        for joint_name, joint_config in self.config["joints"].items():
            if not joint_config["is_passive"]:
                self.init_motor_angles[joint_name] = 0.0

        self.init_joint_angles = self.motor_to_joint_angles(self.init_motor_angles)

        self.motor_ordering = list(self.init_motor_angles.keys())
        self.joint_ordering = list(self.init_joint_angles.keys())

        self.default_motor_angles: Dict[str, float] = {}
        for joint_name, joint_config in self.config["joints"].items():
            if not joint_config["is_passive"]:
                self.default_motor_angles[joint_name] = joint_config["default_pos"]

        self.default_joint_angles = self.motor_to_joint_angles(
            self.default_motor_angles
        )

        joints_config = self.config["joints"]
        self.motor_to_joint_name: Dict[str, List[str]] = {}
        self.joint_to_motor_name: Dict[str, List[str]] = {}
        for motor_name, joint_name in zip(self.motor_ordering, self.joint_ordering):
            transmission = joints_config[motor_name]["transmission"]
            if transmission == "ankle":
                if "left" in motor_name:
                    self.motor_to_joint_name[motor_name] = [
                        "left_ank_roll",
                        "left_ank_pitch",
                    ]
                    self.joint_to_motor_name[joint_name] = [
                        "left_ank_act_1",
                        "left_ank_act_2",
                    ]
                elif "right" in motor_name:
                    self.motor_to_joint_name[motor_name] = [
                        "right_ank_roll",
                        "right_ank_pitch",
                    ]
                    self.joint_to_motor_name[joint_name] = [
                        "right_ank_act_1",
                        "right_ank_act_2",
                    ]
            elif transmission == "waist":
                self.motor_to_joint_name[motor_name] = ["waist_roll", "waist_yaw"]
                self.joint_to_motor_name[joint_name] = ["waist_act_1", "waist_act_2"]
            else:
                self.motor_to_joint_name[motor_name] = [joint_name]
                self.joint_to_motor_name[joint_name] = [motor_name]

        self.passive_joint_names = []
        for joint_name in self.joint_ordering:
            transmission = joints_config[joint_name]["transmission"]
            if transmission == "linkage":
                for suffix in [
                    "_front_rev_1",
                    "_front_rev_2",
                    "_back_rev_1",
                    "_back_rev_2",
                ]:
                    self.passive_joint_names.append(joint_name + suffix)
            elif transmission == "rack_and_pinion":
                self.passive_joint_names.append(joint_name + "_mirror")

        if "foot_name" in self.config["general"]:
            self.foot_name = self.config["general"]["foot_name"]

        self.has_gripper = False
        for motor_name in self.motor_ordering:
            if "gripper" in motor_name:
                self.has_gripper = True

        self.collider_names: List[str] = []
        for link_name, link_config in self.collision_config.items():
            if link_config["has_collision"]:
                self.collider_names.append(link_name)

        self.nu = len(self.motor_ordering)
        self.joint_groups: Dict[str, str] = {}
        for joint_name, joint_config in self.config["joints"].items():
            self.joint_groups[joint_name] = joint_config["group"]

        self.joint_limits: Dict[str, List[float]] = {}
        for joint_name, joint_config in self.config["joints"].items():
            self.joint_limits[joint_name] = [
                joint_config["lower_limit"],
                joint_config["upper_limit"],
            ]

    def get_joint_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str = "name",
        group: str = "all",
    ) -> List[Any]:
        """Returns a list of attributes for joints that match specified criteria.

        Args:
            key_name (str): The key to search for in each joint's configuration.
            key_value (Any): The value that the specified key must have for a joint to be included.
            attr_name (str, optional): The attribute to retrieve from each matching joint. Defaults to "name".
            group (str, optional): The group to which the joint must belong. Use "all" to include joints from any group. Defaults to "all".

        Returns:
            List[Any]: A list of attributes from joints that match the specified key-value pair and group criteria.
        """
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
        """Sets attributes for joints in the configuration based on specified criteria.

        Args:
            key_name (str): The key to match in each joint's configuration.
            key_value (Any): The value that the key must have for the joint to be modified.
            attr_name (str): The name of the attribute to set for matching joints.
            attr_values (Any): The values to assign to the attribute. Can be a dictionary or a list.
            group (str, optional): The group to which the joint must belong to be modified. Defaults to "all".
        """
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
        """Calculates the forward kinematics for the waist joint based on motor positions.

        Args:
            motor_pos (List[float]): A list containing the positions of the motors.

        Returns:
            List[float]: A list containing the calculated waist roll and yaw angles.
        """
        offsets = self.config["general"]["offsets"]
        waist_roll = offsets["waist_roll_coef"] * (-motor_pos[0] + motor_pos[1])
        waist_yaw = offsets["waist_yaw_coef"] * (motor_pos[0] + motor_pos[1])
        return [waist_roll, waist_yaw]

    def waist_ik(self, waist_pos: List[float]) -> List[float]:
        """Calculates the inverse kinematics for the waist actuators based on the desired waist position.

        Args:
            waist_pos (List[float]): A list containing the desired roll and yaw positions of the waist.

        Returns:
            List[float]: A list containing the calculated positions for the two waist actuators.
        """
        offsets = self.config["general"]["offsets"]
        roll = waist_pos[0] / offsets["waist_roll_coef"]
        yaw = waist_pos[1] / offsets["waist_yaw_coef"]
        waist_act_1 = (-roll + yaw) / 2
        waist_act_2 = (roll + yaw) / 2
        return [waist_act_1, waist_act_2]

    # @profile()
    def motor_to_joint_angles(self, motor_angles: Dict[str, float]) -> Dict[str, float]:
        """Converts motor angles to joint angles based on the robot's configuration.

        Args:
            motor_angles (Dict[str, float]): A dictionary mapping motor names to their respective angles.

        Returns:
            Dict[str, float]: A dictionary mapping joint names to their calculated angles.
        """
        joint_angles: Dict[str, float] = {}
        joints_config = self.config["joints"]
        waist_act_pos: List[float] = []
        left_ank_act_pos: List[float] = []
        right_ank_act_pos: List[float] = []
        for motor_name, motor_pos in motor_angles.items():
            transmission = joints_config[motor_name]["transmission"]
            if transmission == "gear":
                joint_name = motor_name.replace("_drive", "_driven")
                joint_angles[joint_name] = (
                    -motor_pos * joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "rack_and_pinion":
                joint_pinion_name = motor_name.replace("_rack", "_pinion")
                joint_angles[joint_pinion_name] = (
                    -motor_pos * joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "waist":
                # Placeholder to ensure the correct order
                joint_angles["waist_roll"] = 0.0
                joint_angles["waist_yaw"] = 0.0
                waist_act_pos.append(motor_pos)
            elif transmission == "linkage":
                joint_angles[motor_name.replace("_act", "")] = motor_pos
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

        return joint_angles

    def joint_to_motor_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        """Converts joint angles to motor angles based on the transmission type specified in the configuration.

        Args:
            joint_angles (Dict[str, float]): A dictionary mapping joint names to their respective angles.

        Returns:
            Dict[str, float]: A dictionary mapping motor names to their calculated angles.
        """
        motor_angles: Dict[str, float] = {}
        joints_config = self.config["joints"]
        waist_pos: List[float] = []
        left_ankle_pos: List[float] = []
        right_ankle_pos: List[float] = []
        for joint_name, joint_pos in joint_angles.items():
            transmission = joints_config[joint_name]["transmission"]
            if transmission == "gear":
                motor_name = joint_name.replace("_driven", "_drive")
                motor_angles[motor_name] = (
                    -joint_pos / joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "rack_and_pinion":
                motor_name = joint_name.replace("_pinion", "_rack")
                motor_angles[motor_name] = (
                    -joint_pos / joints_config[motor_name]["gear_ratio"]
                )
            elif transmission == "waist":
                # Placeholder to ensure the correct order
                motor_angles["waist_act_1"] = 0.0
                motor_angles["waist_act_2"] = 0.0
                waist_pos.append(joint_pos)
            elif transmission == "linkage":
                motor_angles[joint_name + "_act"] = joint_pos
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

        if len(waist_pos) > 0:
            motor_angles["waist_act_1"], motor_angles["waist_act_2"] = self.waist_ik(
                waist_pos
            )

        return motor_angles

    def joint_to_passive_angles(
        self, joint_angles: Dict[str, float]
    ) -> Dict[str, float]:
        """Converts joint angles to passive angles based on the transmission type.

        This function processes a dictionary of joint angles and converts them into passive angles using the transmission configuration specified in the object's configuration. It supports two types of transmissions: 'linkage' and 'rack_and_pinion'. For 'linkage' transmissions, it generates additional passive angles with specific suffixes, applying a sign change for knee joints. For 'rack_and_pinion' transmissions, it mirrors the joint angle.

        Args:
            joint_angles (Dict[str, float]): A dictionary where keys are joint names and values are their respective angles.

        Returns:
            Dict[str, float]: A dictionary of passive angles derived from the input joint angles.
        """
        passive_angles: Dict[str, float] = {}
        joints_config = self.config["joints"]
        for joint_name, joint_pos in joint_angles.items():
            transmission = joints_config[joint_name]["transmission"]
            if transmission == "linkage":
                sign = 1 if "knee" in joint_name else -1
                for suffix in [
                    "_front_rev_1",
                    "_front_rev_2",
                    "_back_rev_1",
                    "_back_rev_2",
                ]:
                    passive_angles[joint_name + suffix] = sign * joint_pos
            elif transmission == "rack_and_pinion":
                passive_angles[joint_name + "_mirror"] = joint_pos

        return passive_angles
