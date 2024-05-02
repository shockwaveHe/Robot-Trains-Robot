import argparse
import os
import xml.etree.ElementTree as ET

from jinja2 import Environment, FileSystemLoader

from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.misc_utils import snake2camel

rewards_config = {
    "base_height_target": 0.3,
    "min_dist": 0.06,
    "max_dist": 0.15,
    "target_joint_pos_scale": 0.17,  # rad
    "target_feet_height": 0.04,  # m
    "cycle_time": 0.64,  # sec
    "only_positive_rewards": True,  # if true negative total rewards are clipped at zero (avoids early termination problems)
    "tracking_sigma": 5,  # tracking reward = exp(error*sigma)
    "max_contact_force": 50,  # Maximum allowable contact force
}

reward_scales_config = {
    "joint_pos": 1.6,
    "feet_clearance": 1.0,
    "feet_contact_number": 1.2,
    "feet_air_time": 1.0,
    "foot_slip": -0.05,
    "feet_distance": 0.2,
    "knee_distance": 0.2,
    "feet_contact_forces": -0.01,
    "tracking_lin_vel": 1.2,
    "tracking_ang_vel": 1.1,
    "vel_mismatch_exp": 0.5,  # lin_z; ang x, y
    "low_speed": 0.2,
    "track_vel_hard": 0.5,
    "default_joint_pos": 0.5,
    "orientation": 1.0,
    "base_height": 0.2,
    "base_acc": 0.2,
    "action_smoothness": -0.002,
    "torques": -1e-5,
    "dof_vel": -5e-4,
    "dof_acc": -1e-7,
    "collision": -1.0,
}


def create_isaac_urdf(robot_name):
    robot = HumanoidRobot(robot_name)
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    source_urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")
    tree = ET.parse(source_urdf_path)
    root = tree.getroot()

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name not in robot.config.motor_params and joint_type != "fixed":
            joint.set("type", "fixed")  # Set joint type to 'fixed'

        for jp in joint.findall("joint_properties"):
            joint.remove(jp)

        # Update the effort in <limit>
        limit = joint.find("limit")
        if limit is not None:
            limit.set("effort", "10")  # Update the effort attribute to 10
            limit.set("velocity", "20")  # Update the velocity attribute to 20

        # Add <dynamics> element
        joint_name = joint.get("name")
        if joint_name in robot.config.motor_params:
            motor_params = robot.config.motor_params[joint_name]
            dynamics = ET.Element("dynamics")
            dynamics.set("damping", str(motor_params.damping))
            joint.append(dynamics)

    isaac_urdf_path = os.path.join(robot_dir, f"{robot_name}_isaac.urdf")

    tree.write(isaac_urdf_path)


def create_humanoid_gym_env(robot_name):
    # check if the humanoid_gym submodule is present
    humanoid_gym_root_dir = os.path.join("toddlerbot", "sim", "humanoid_gym")
    humanoid_gym_env_dir = os.path.join(humanoid_gym_root_dir, "humanoid", "envs")
    if not os.path.exists(humanoid_gym_env_dir):
        raise FileNotFoundError(
            f"The directory {humanoid_gym_env_dir} was not found.\n"
            + "Please run git submodule update --init."
        )

    env_path = os.path.join(humanoid_gym_env_dir, f"{robot_name}")
    os.makedirs(env_path, exist_ok=True)

    robot = HumanoidRobot(robot_name)
    num_actions = len(robot.config.motor_params)
    foot_name = robot.config.canonical_name2link_name["foot_link"]
    knee_name = robot.config.canonical_name2link_name["knee_link"]

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    isaac_urdf_path = os.path.join(robot_dir, f"{robot_name}_isaac.urdf")
    tree = ET.parse(isaac_urdf_path)
    root = tree.getroot()

    terminate_after_contacts_on = []
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if foot_name not in link_name and link.find("collision") is not None:
            terminate_after_contacts_on.append(link_name)

    template_dir = os.path.join("toddlerbot", "sim", "humanoid_gym", "humanoid", "envs")
    config_template_name = "humanoid_config_template.py.j2"
    template_env = Environment(loader=FileSystemLoader(template_dir))
    config_template = template_env.get_template(config_template_name)

    robot_name_camel = snake2camel(robot_name)
    urdf_relpath = os.path.relpath(isaac_urdf_path, humanoid_gym_root_dir)

    config_script_content = config_template.render(
        robot_name=robot_name,
        robot_name_camel=robot_name_camel,
        num_actions=num_actions,
        urdf_relpath=urdf_relpath,
        foot_name=foot_name,
        knee_name=knee_name,
        terminate_after_contacts_on=terminate_after_contacts_on,
        default_joint_angles={
            key: params.default_angle
            for key, params in robot.config.motor_params.items()
        },
        motor_kp_dict={
            key: params.kp for key, params in robot.config.motor_params.items()
        },
        motor_kv_dict={
            key: params.kv for key, params in robot.config.motor_params.items()
        },
        rewards=rewards_config,
        reward_scales=reward_scales_config,
        experiment_name=f"walk_{robot_name}_isaac",
        run_name="v0.1",
    )

    config_script_path = os.path.join(env_path, f"{robot_name}_config.py")
    with open(config_script_path, "w") as file:
        file.write(config_script_content)

    # Format the import and registration lines
    env_class = f"{robot_name_camel}Env"
    cfg_class = f"{robot_name_camel}Cfg"
    ppo_class = f"{robot_name_camel}CfgPPO"
    import_lines = (
        f"from .{robot_name}.{robot_name}_env import {env_class}\n"
        f"from .{robot_name}.{robot_name}_config import {cfg_class}, {ppo_class}\n"
    )

    registration_line = f'task_registry.register( "{robot_name}", {env_class}, {cfg_class}(), {ppo_class}() )\n'

    register_script_path = os.path.join(humanoid_gym_env_dir, "__init__.py")
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

    create_isaac_urdf(args.robot_name)
    create_humanoid_gym_env(args.robot_name)
