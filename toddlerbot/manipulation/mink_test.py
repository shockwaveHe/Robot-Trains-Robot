import argparse

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import yaml
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from toddlerbot.manipulation.teleoperation.data_processing.ip_config import *
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib import (
    mink_retarget,
)
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib.utils.draw import (
    draw_frame,
)


def yaml_table_2_dict(yaml_table):
    """
    Convert a yaml table to a dictionary
    """
    yaml_dict = {}
    for key, value in yaml_table.items():
        if key in yaml_dict.keys():
            for sub_key, sub_value in value.items():
                yaml_dict[key].append(sub_value)
        else:
            yaml_dict[key] = []
            for sub_key, sub_value in value.items():
                yaml_dict[key].append(sub_value)
    return yaml_dict

if __name__ == "__main__":
    # create a mujoco word loading the toddlerbot model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view_frame",
        default=False,
        help="Visual frames in mujoco",
    )
    args = parser.parse_args()

    xml_path = "..\\..\\toddlerbot\\descriptions\\toddlerbot_active\\toddlerbot_active_scene_mink_test.xml"
    # ik_config = "..\\..\\toddlerbot\\manipulation\\teleoperation\\ik_configs\\quest_toddy.json"
    ik_config = "..\\..\\toddlerbot\\manipulation\\teleoperation\\ik_configs\\quest_toddy_mink_test.yaml"  # Switch to yaml for easy debug
    model = mj.MjModel.from_xml_path(xml_path)
    model.opt.gravity[:] = np.array([0, 0, 0])
    # Create a data structure for simulation
    data = mj.MjData(model)
    # Load the IK config
    with open(ik_config) as f:
        # ik_config = json.load(f)
        ik_config = yaml.safe_load(f)
    
    # ik_match_table = ik_config.pop("ik_match_table")
    # the retarget library requires a dictionary with the following structure:
    ik_match_table = yaml_table_2_dict(ik_config["ik_match_table"])
    retarget = mink_retarget.MinkRetarget(
        xml_path,
        ik_match_table,
        scale=ik_config["scale"],
        ground=ik_config["ground_height"],
    )
    anchor_list = ik_config.pop("anchor_list")

    # Launch the MuJoCo viewer
    with mjv.launch_passive(model, data) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE
        rate = RateLimiter(frequency=50.0, warn=False)
        t = 0
        dt = rate.dt
        retarget.configuration.update_from_keyframe("home")
        initial_pose = {}
        initial_rot = {}
        # get initial position for anchor joints
        
        for robot_link, ik_data in ik_match_table.items():
            if robot_link in anchor_list:
                mid = model.body(ik_data[0]).mocapid[0] 
                initial_pose[robot_link] = retarget.configuration.get_transform_frame_to_world(robot_link, ik_data[1]).translation()
                if robot_link in ['head', 'torso']:
                    initial_pose[robot_link][2] += .5
                # if robot_link in ["torso", "waist_gears"]:
                #     initial_pose[robot_link][2] += 0.015
                wxyz = retarget.configuration.get_transform_frame_to_world(robot_link, ik_data[1]).rotation().wxyz
                data.mocap_pos[mid] = initial_pose[robot_link]
                data.mocap_quat[mid] = wxyz
                xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
                initial_rot[robot_link] = xyzw
            elif robot_link == "left_ee_center":
                wxyz = retarget.configuration.get_transform_frame_to_world(robot_link, ik_data[1]).rotation().wxyz
                initial_rot[robot_link] = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]) # xyzw

        print("initial_head_rot: ", initial_rot["head"])

        
        gripper_target_mid = model.body("left_hand_pose").mocapid[0]

        while viewer.is_running():
            
            data.mocap_pos[gripper_target_mid][1] += 0.0005 * np.cos(0.5 * t + np.pi)
            data.mocap_pos[gripper_target_mid][2] += 0.0005 * np.sin(0.5 * t)
            vicon_data = {}
            
            for robot_link, ik_data in ik_match_table.items():
                # robot_link of mujoco, ik_data[0] from vicon
                if robot_link in anchor_list:
                    vicon_data[ik_data[0]] = [initial_pose[robot_link], initial_rot[robot_link]]
                elif robot_link == "left_ee_center":
                    target_pose = data.mocap_pos[gripper_target_mid]
                    # target_rot_wxyz = data.mocap_quat[gripper_target_mid]
                    # target_rot_xyzw = np.array([target_rot_wxyz[1], target_rot_wxyz[2], target_rot_wxyz[3], target_rot_wxyz[0]])
                    target_rot_xyzw = initial_rot["left_ee_center"]
                    vicon_data[ik_data[0]] = [target_pose, target_rot_xyzw] 

            # Draw the task targets for reference
            for robot_link, ik_data in ik_match_table.items():
                if ik_data[0] not in vicon_data:
                    continue
                if args.view_frame:
                    draw_frame(
                        ik_config["scale"] * np.array(vicon_data[ik_data[0]][0])
                        - retarget.ground,
                        R.from_quat(
                            vicon_data[ik_data[0]][1], scalar_first=True
                        ).as_matrix(),
                        viewer,
                        0.1,
                        orientation_correction=R.from_quat(ik_data[-1], scalar_first=True),
                    )
            # Retarget and pose the model
            retarget.update_targets(vicon_data)
            # data.qpos[:] = retarget.retarget(vicon_data).copy()
            data.qpos[:] = retarget.retarget_v2(vicon_data, dt=rate.dt, max_iters=20).copy() # better implementation for retargeting performance
            mj.mj_forward(model, data)
            # retarget.configuration.data = data
            # Visualize at fixed FPS.
            viewer.sync()
            t += dt
            rate.sleep()