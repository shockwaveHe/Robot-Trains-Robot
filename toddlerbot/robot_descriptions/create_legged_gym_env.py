import argparse
import os
import textwrap
import xml.etree.ElementTree as ET

from toddlerbot.sim.robot import HumanoidRobot


def process_urdf_isaac(robot):
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot.name)
    source_urdf_path = os.path.join(robot_dir, f"{robot.name}.urdf")
    issac_urdf_path = os.path.join(robot_dir, f"{robot.name}_isaac.urdf")
    tree = ET.parse(source_urdf_path)
    root = tree.getroot()

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name not in robot.config.motor_params and joint_type != "fixed":
            joint.set("type", "fixed")  # Set joint type to 'fixed'

    tree.write(issac_urdf_path)

    return issac_urdf_path


def create_legged_gym_env(robot_name):
    robot = HumanoidRobot(robot_name)
    num_actions = len(robot.config.motor_params)

    legged_gym_root_dir = os.path.join("toddlerbot", "sim", "legged_gym")
    legged_gym_env_dir = os.path.join(legged_gym_root_dir, "legged_gym", "envs")
    if not os.path.exists(legged_gym_env_dir):
        raise FileNotFoundError(
            f"The directory {legged_gym_env_dir} was not found.\nPlease run git submodule update --init."
        )

    env_path = os.path.join(legged_gym_env_dir, f"{robot_name}")
    os.makedirs(env_path, exist_ok=True)

    isaac_urdf_path = process_urdf_isaac(robot)
    urdf_relpath = os.path.relpath(isaac_urdf_path, legged_gym_root_dir)

    robot_name_capitalized = robot_name.capitalize()

    config_script_content = textwrap.dedent(
        f"""\
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


    class {robot_name_capitalized}Cfg(LeggedRobotCfg):
        class env(LeggedRobotCfg.env):
            num_envs = 4096
            num_observations = {num_actions * 3 + 3 * 4 + 187}
            num_actions = {num_actions}

        class terrain(LeggedRobotCfg.terrain):
            mesh_type = "plane"

        class init_state(LeggedRobotCfg.init_state):
            pos = [0.0, 0.0, 1.0]  # x,y,z [m]
            default_joint_angles = {{  # = target angles [rad] when action = 0.0
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.325,
                "left_knee": 0.65,
                "left_ank_pitch": 0.325,
                "left_ank_roll": 0.0,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": -0.325,
                "right_knee": -0.65,
                "right_ank_pitch": -0.325,
                "right_ank_roll": 0.0,
                "left_sho_pitch": 0.0,
                "left_sho_roll": -1.57,
                "left_elb": 0.0,
                "right_sho_pitch": 0.0,
                "right_sho_roll": 1.57,
                "right_elb": 0.0,
            }}

        class control(LeggedRobotCfg.control):
            # PD Drive parameters:
            stiffness = {{   
                "left_hip_yaw": 100.0,
                "left_hip_roll": 100.0,
                "left_hip_pitch": 100.0,
                "left_knee": 100.0,
                "left_ank_pitch": 100.0,
                "left_ank_roll": 100.0,
                "right_hip_yaw": 100.0,
                "right_hip_roll": 100.0,
                "right_hip_pitch": 100.0,
                "right_knee": 100.0,
                "right_ank_pitch": 100.0,
                "right_ank_roll": 100.0,
                "left_sho_pitch": 100.0,
                "left_sho_roll": 100.0,
                "left_elb": 100.0,
                "right_sho_pitch": 100.0,
                "right_sho_roll": 100.0,
                "right_elb": 100.0,
            }}  # [N*m/rad]
            damping = {{ 
                "left_hip_yaw": 10.0,
                "left_hip_roll": 10.0,
                "left_hip_pitch": 10.0,
                "left_knee": 10.0,
                "left_ank_pitch": 10.0,
                "left_ank_roll": 10.0,
                "right_hip_yaw": 10.0,
                "right_hip_roll": 10.0,
                "right_hip_pitch": 10.0,
                "right_knee": 10.0,
                "right_ank_pitch": 10.0,
                "right_ank_roll": 10.0,
                "left_sho_pitch": 10.0,
                "left_sho_roll": 10.0,
                "left_elb": 10.0,
                "right_sho_pitch": 10.0,
                "right_sho_roll": 10.0,
                "right_elb": 10.0,
            }}  # [N*m*s/rad]
            # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.5
            # decimation: Number of control action updates @ sim DT per policy DT
            decimation = 4
            
        class asset(LeggedRobotCfg.asset):
            file = "{{LEGGED_GYM_ROOT_DIR}}/{urdf_relpath}"
            name = "{robot_name}"
            foot_name = "ank_roll_link"
            terminate_after_contacts_on = ["body_link"]
            flip_visual_attachments = False
            self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

        class rewards(LeggedRobotCfg.rewards):
            soft_dof_pos_limit = 0.95
            soft_dof_vel_limit = 0.9
            soft_torque_limit = 0.9
            max_contact_force = 300.0
            only_positive_rewards = False

            class scales(LeggedRobotCfg.rewards.scales):
                termination = -200.0
                tracking_ang_vel = 1.0
                torques = -5.0e-6
                dof_acc = -2.0e-7
                lin_vel_z = -0.5
                feet_air_time = 5.0
                dof_pos_limits = -1.0
                no_fly = 0.25
                dof_vel = -0.0
                ang_vel_xy = -0.0
                feet_contact_forces = -0.0


    class {robot_name_capitalized}CfgPPO(LeggedRobotCfgPPO):

        class runner(LeggedRobotCfgPPO.runner):
            run_name = ""
            experiment_name = "walk_{robot_name}"

        class algorithm(LeggedRobotCfgPPO.algorithm):
            entropy_coef = 0.01
    """
    )

    config_script_path = os.path.join(env_path, f"{robot_name}_config.py")
    with open(config_script_path, "w") as file:
        file.write(config_script_content)

    script_content = textwrap.dedent(
        f"""\
    import os
    from time import time
    from typing import Dict, Tuple

    import numpy as np
    import torch
    from isaacgym import gymapi, gymtorch, gymutil
    from isaacgym.torch_utils import *
    from legged_gym.envs import LeggedRobot


    class {robot_name_capitalized}(LeggedRobot):
        def _reward_no_fly(self):
            contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
            single_contact = torch.sum(1.0 * contacts, dim=1) == 1
            return 1.0 * single_contact
    """
    )

    script_path = os.path.join(env_path, f"{robot_name}.py")
    with open(script_path, "w") as file:
        file.write(script_content)

    # Format the import and registration lines
    cfg_class = f"{robot_name_capitalized}Cfg"
    ppo_class = f"{robot_name_capitalized}CfgPPO"
    import_lines = (
        f"from .{robot_name}.{robot_name} import {robot_name_capitalized}\n"
        f"from .{robot_name}.{robot_name}_config import {cfg_class}, {ppo_class}\n"
    )

    registration_line = f'task_registry.register( "{robot_name}", {robot_name_capitalized}, {cfg_class}(), {ppo_class}() )\n'

    register_script_path = os.path.join(legged_gym_env_dir, "__init__.py")
    with open(register_script_path, "r") as file:
        content = file.readlines()

    content_str = "".join(content)

    # Check if the robot is already registered or imported
    if (f".{robot_name}.{robot_name}" in content_str) or (
        f'task_registry.register("{robot_name}"' in content_str
    ):
        print(f"{robot_name} appears to be already registered.")
    else:
        last_import_index = max(
            idx for idx, line in enumerate(content) if "import" in line
        )
        content.insert(last_import_index + 1, import_lines)
        content.append(registration_line)
        with open(register_script_path, "w") as file:
            file.writelines(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the collisions.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    create_legged_gym_env(args.robot_name)
