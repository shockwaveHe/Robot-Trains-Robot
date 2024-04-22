import argparse
import os
import textwrap
import xml.etree.ElementTree as ET

from toddlerbot.sim.robot import HumanoidRobot

MAX_ITERATIONS = 1500


def create_humanoid_gym_env(robot_name):
    humanoid_gym_root_dir = os.path.join("toddlerbot", "sim", "humanoid_gym")
    humanoid_gym_env_dir = os.path.join(humanoid_gym_root_dir, "humanoid", "envs")
    if not os.path.exists(humanoid_gym_env_dir):
        raise FileNotFoundError(
            f"The directory {humanoid_gym_env_dir} was not found.\nPlease run git submodule update --init."
        )

    env_path = os.path.join(humanoid_gym_env_dir, f"{robot_name}")
    os.makedirs(env_path, exist_ok=True)

    robot = HumanoidRobot(robot_name)
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    source_urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")
    tree = ET.parse(source_urdf_path)
    root = tree.getroot()

    num_actions = len(robot.config.motor_params)
    foot_name = robot.config.canonical_name2link_name["foot_link"]
    knee_name = robot.config.canonical_name2link_name["knee_link"]

    terminate_after_contacts_on = []
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if foot_name not in link_name and link.find("collision") is not None:
            terminate_after_contacts_on.append(link_name)

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name not in robot.config.motor_params and joint_type != "fixed":
            joint.set("type", "fixed")  # Set joint type to 'fixed'

    isaac_urdf_path = os.path.join(robot_dir, f"{robot_name}_isaac.urdf")
    tree.write(isaac_urdf_path)

    urdf_relpath = os.path.relpath(isaac_urdf_path, humanoid_gym_root_dir)

    robot_name_capitalized = "".join(x.title() for x in robot_name.split("_"))

    config_script_content = textwrap.dedent(
        f"""\
    from humanoid_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


    class {robot_name_capitalized}Cfg(LeggedRobotCfg):
        class env(LeggedRobotCfg.env):
            frame_stack = 15
            c_frame_stack = 3
            num_single_obs = 47
            num_observations = int(frame_stack * num_single_obs)
            single_num_privileged_obs = 73
            num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
            num_actions = {num_actions}
            num_envs = 4096
            episode_length_s = 24  # episode length in seconds
            use_ref_actions = False

        class safety:
            # safety factors
            pos_limit = 1.0
            vel_limit = 1.0
            torque_limit = 0.85

        
        class asset(LeggedRobotCfg.asset):
            file = "{{LEGGED_GYM_ROOT_DIR}}/{urdf_relpath}"
            
            name = "{robot_name}"
            foot_name = "{foot_name}"
            knee_name = "{knee_name}"
            
            terminate_after_contacts_on = {terminate_after_contacts_on}
            penalize_contacts_on = {terminate_after_contacts_on}
            self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
            flip_visual_attachments = False
            replace_cylinder_with_capsule = False
            fix_base_link = False

        class terrain(LeggedRobotCfg.terrain):
            mesh_type = 'plane'
            # mesh_type = 'trimesh'
            curriculum = False
            # rough terrain only:
            measure_heights = False
            static_friction = 0.6
            dynamic_friction = 0.6
            terrain_length = 8.
            terrain_width = 8.
            num_rows = 20  # number of terrain rows (levels)
            num_cols = 20  # number of terrain cols (types)
            max_init_terrain_level = 10  # starting curriculum state
            # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
            terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
            restitution = 0.

        class noise:
            add_noise = True
            noise_level = 0.6    # scales other values

            class noise_scales:
                dof_pos = 0.05
                dof_vel = 0.5
                ang_vel = 0.1
                lin_vel = 0.05
                quat = 0.03
                height_measurements = 0.1

        class init_state(LeggedRobotCfg.init_state):
            pos = [0.0, 0.0, 0.0]  # x,y,z [m]
            
            default_joint_angles = {{  # = target angles [rad] when action = 0.0
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.0,
                "left_knee": 0.0,
                "left_ank_pitch": 0.0,
                "left_ank_roll": 0.0,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": -0.0,
                "right_knee": -0.0,
                "right_ank_pitch": -0.0,
                "right_ank_roll": 0.0,
                # "left_sho_pitch": 0.0,
                # "left_sho_roll": -1.57,
                # "left_elb": 0.0,
                # "right_sho_pitch": 0.0,
                # "right_sho_roll": 1.57,
                # "right_elb": 0.0,
            }}

        class control(LeggedRobotCfg.control):
            # PD Drive parameters:
            stiffness = {{   
                "left_hip_yaw": {robot.config.motor_params["left_hip_yaw"].kp},
                "left_hip_roll": {robot.config.motor_params["left_hip_roll"].kp},
                "left_hip_pitch": {robot.config.motor_params["left_hip_pitch"].kp},
                "left_knee": {robot.config.motor_params["left_knee"].kp},
                "left_ank_pitch": {robot.config.motor_params["left_ank_pitch"].kp},
                "left_ank_roll": {robot.config.motor_params["left_ank_roll"].kp},
                "right_hip_yaw": {robot.config.motor_params["right_hip_yaw"].kp},
                "right_hip_roll": {robot.config.motor_params["right_hip_roll"].kp},
                "right_hip_pitch":{robot.config.motor_params["right_hip_pitch"].kp},
                "right_knee": {robot.config.motor_params["right_knee"].kp},
                "right_ank_pitch": {robot.config.motor_params["right_ank_pitch"].kp},
                "right_ank_roll": {robot.config.motor_params["right_ank_roll"].kp},
                # "left_sho_pitch": 100.0,
                # "left_sho_roll": 100.0,
                # "left_elb": 100.0,
                # "right_sho_pitch": 100.0,
                # "right_sho_roll": 100.0,
                # "right_elb": 100.0,
            }}  # [N*m/rad]
            damping = {{ 
                "left_hip_yaw": {robot.config.motor_params["left_hip_yaw"].damping},
                "left_hip_roll": {robot.config.motor_params["left_hip_roll"].damping},
                "left_hip_pitch": {robot.config.motor_params["left_hip_pitch"].damping},
                "left_knee": {robot.config.motor_params["left_knee"].damping},
                "left_ank_pitch": {robot.config.motor_params["left_ank_pitch"].damping},
                "left_ank_roll": {robot.config.motor_params["left_ank_roll"].damping},
                "right_hip_yaw": {robot.config.motor_params["right_hip_yaw"].damping},
                "right_hip_roll": {robot.config.motor_params["right_hip_roll"].damping},
                "right_hip_pitch":{robot.config.motor_params["right_hip_pitch"].damping},
                "right_knee": {robot.config.motor_params["right_knee"].damping},
                "right_ank_pitch": {robot.config.motor_params["right_ank_pitch"].damping},
                "right_ank_roll": {robot.config.motor_params["right_ank_roll"].damping},
                # "left_sho_pitch": 4.0,
                # "left_sho_roll": 4.0,
                # "left_elb": 4.0,
                # "right_sho_pitch": 4.0,
                # "right_sho_roll": 4.0,
                # "right_elb": 4.0,
            }}  # [N*m*s/rad]
            # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.25
            # decimation: Number of control action updates @ sim DT per policy DT
            decimation = 10  # 100hz
            
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

        class domain_rand:
            randomize_friction = True
            friction_range = [0.1, 2.0]
            randomize_base_mass = True
            added_mass_range = [-5., 5.]
            push_robots = True
            push_interval_s = 4
            max_push_vel_xy = 0.2
            max_push_ang_vel = 0.4
            dynamic_randomization = 0.02

        class commands(LeggedRobotCfg.commands):
            # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            num_commands = 4
            resampling_time = 8.  # time before command are changed[s]
            heading_command = True  # if true: compute ang vel command from heading error

            class ranges:
                lin_vel_x = [-0.3, 0.6]  # min max [m/s]
                lin_vel_y = [-0.3, 0.3]   # min max [m/s]
                ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
                heading = [-3.14, 3.14]

        class rewards(LeggedRobotCfg.rewards):
            base_height_target = 0.89
            min_dist = 0.2
            max_dist = 0.5
            # put some settings here for LLM parameter tuning
            target_joint_pos_scale = 0.17    # rad
            target_feet_height = 0.06       # m
            cycle_time = 0.64                # sec
            # if true negative total rewards are clipped at zero (avoids early termination problems)
            only_positive_rewards = True
            # tracking reward = exp(error*sigma)
            tracking_sigma = 5
            max_contact_force = 700  # forces above this value are penalized

            class scales:
                # reference motion tracking
                joint_pos = 1.6
                feet_clearance = 1.
                feet_contact_number = 1.2
                # gait
                feet_air_time = 1.
                foot_slip = -0.05
                feet_distance = 0.2
                knee_distance = 0.2
                # contact
                feet_contact_forces = -0.01
                # vel tracking
                tracking_lin_vel = 1.2
                tracking_ang_vel = 1.1
                vel_mismatch_exp = 0.5  # lin_z; ang x,y
                low_speed = 0.2
                track_vel_hard = 0.5
                # base pos
                default_joint_pos = 0.5
                orientation = 1.
                base_height = 0.2
                base_acc = 0.2
                # energy
                action_smoothness = -0.002
                torques = -1e-5
                dof_vel = -5e-4
                dof_acc = -1e-7
                collision = -1.

        class normalization:
            class obs_scales:
                lin_vel = 2.
                ang_vel = 1.
                dof_pos = 1.
                dof_vel = 0.05
                quat = 1.
                height_measurements = 5.0
            clip_observations = 18.
            clip_actions = 18.


    class {robot_name_capitalized}CfgPPO(LeggedRobotCfgPPO):
        seed = 5
        runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

        class policy:
            init_noise_std = 1.0
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [768, 256, 128]
       
        class algorithm(LeggedRobotCfgPPO.algorithm):
            entropy_coef = 0.001
            learning_rate = 1e-5
            num_learning_epochs = 2
            gamma = 0.994
            lam = 0.9
            num_mini_batches = 4

        class runner:
            policy_class_name = 'ActorCritic'
            algorithm_class_name = 'PPO'
            num_steps_per_env = 60  # per iteration
            max_iterations = 3001  # number of policy updates

            # logging
            save_interval = 100  # check for potential saves every this many iterations
            experiment_name = 'XBot_ppo'
            run_name = ''
            # load and resume
            resume = False
            load_run = -1  # -1 = last run
            checkpoint = -1  # -1 = last saved model
            resume_path = None  # updated from load_run and chkpt

    """
    )

    config_script_path = os.path.join(env_path, f"{robot_name}_config.py")
    with open(config_script_path, "w") as file:
        file.write(config_script_content)

    # Format the import and registration lines
    env_class = f"{robot_name_capitalized}Env"
    cfg_class = f"{robot_name_capitalized}Cfg"
    ppo_class = f"{robot_name_capitalized}CfgPPO"
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

    create_humanoid_gym_env(args.robot_name)
