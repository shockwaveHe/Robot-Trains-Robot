import argparse
import os
import textwrap


def create_legged_gym_env(robot_name):
    legged_gym_root_dir = os.path.join("toddlerbot", "sim", "legged_gym")
    legged_gym_env_dir = os.path.join(legged_gym_root_dir, "legged_gym", "envs")
    if not os.path.exists(legged_gym_env_dir):
        raise FileNotFoundError(
            f"The directory {legged_gym_env_dir} was not found.\nPlease run git submodule update --init."
        )

    env_path = os.path.join(legged_gym_env_dir, f"{robot_name}")
    os.makedirs(env_path, exist_ok=True)

    robot_name_capitalized = robot_name.capitalize()

    config_script_content = textwrap.dedent(
        f"""\
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


    class {robot_name_capitalized}Cfg(LeggedRobotCfg):
        class env(LeggedRobotCfg.env):
            num_envs = 4096
            num_observations = 169
            num_actions = 12

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
            file = "{{LEGGED_GYM_ROOT_DIR}}/resources/robots/{robot_name}/urdf/{robot_name}.urdf"
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

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")

    source_path = os.path.abspath(urdf_path)

    # Formulate the destination path
    dest_path = os.path.join(
        legged_gym_root_dir,
        f"resources",
        "robots",
        robot_name,
        "urdf",
        f"{robot_name}.urdf",
    )

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Check if the symbolic link or file already exists, if so, remove it before creating a new link
    if os.path.exists(dest_path) or os.path.islink(dest_path):
        os.remove(dest_path)

    os.symlink(source_path, dest_path)


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
