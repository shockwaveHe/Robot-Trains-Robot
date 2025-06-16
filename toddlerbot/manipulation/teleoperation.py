import argparse
import json
import os
import pathlib
import socket
import time

import mujoco as mj
import mujoco.viewer
import numpy as np
import pybullet as pb
import yaml
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from toddlerbot.manipulation.teleoperation.data_processing import toddy_quest_module
from toddlerbot.manipulation.teleoperation.data_processing.ip_config import *
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib import (
    mink_retarget,
)
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib.utils.draw import (
    draw_frame,
    draw_frame_batch,
)
from toddlerbot.manipulation.teleoperation.data_processing.rigid_body_sento import (
    create_primitive_shape,
)
from toddlerbot.manipulation.utils.teleop_utils import *


def is_touch(pos, x_touch_range, y_touch_range):
    return (
        x_touch_range[0] < pos[0] < x_touch_range[1]
        and y_touch_range[0] < pos[1] < y_touch_range[1]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view_frame",
        action="store_true",
        help="Visual frames in mujoco",
    )
    args = parser.parse_args()
    # create a mujoco word loading the toddlerbot model
    if os.name == "nt":
        xml_path = "toddlerbot\\descriptions\\toddlerbot_active\\toddlerbot_active_scene_teleop.xml"
        ik_config = "toddlerbot\\manipulation\\teleoperation\\ik_configs\\quest_toddy.yaml"  # Switch to yaml for easy debug
    elif os.name == "posix":
        xml_path = "toddlerbot/descriptions/toddlerbot_active/toddlerbot_active_scene_teleop.xml"
        # xml_path = "toddlerbot/descriptions/toddlerbot/toddlerbot_scene_teleop.xml"
        ik_config = "toddlerbot/manipulation/teleoperation/ik_configs/quest_toddy.yaml"
    else:
        raise Exception("Unsupported OS")
    model = mj.MjModel.from_xml_path(xml_path)
    model.opt.gravity[:] = np.array([0, 0, 0])
    # Create a data structure for simulation
    data = mj.MjData(model)

    # initialize the quest robot module
    c = pb.connect(pb.DIRECT)
    vis_sp = []
    c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
    for i in range(4):
        vis_sp.append(
            create_primitive_shape(pb, 0.1, pb.GEOM_SPHERE, [0.02], color=c_code[i])
        )
    quest = toddy_quest_module.ToddyQuestBimanualModule(
        VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=vis_sp
    )

    # create mink model
    # Load the IK config
    with open(ik_config) as f:
        # ik_config = json.load(f)
        ik_config = yaml.safe_load(f)
    ik_match_table = yaml_table_2_dict(ik_config["ik_match_table"])
    retarget = mink_retarget.MinkRetarget(
        xml_path,
        ik_match_table,
        scale=ik_config["scale"],
        ground=ik_config["ground_height"],
    )
    anchor_list = ik_config.pop("anchor_list")
    safety_constrainted_link_list = ik_config["safety_constraints"]["safety_list"]

    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        rate = RateLimiter(frequency=50.0, warn=False)
        t = 0
        dt = rate.dt
        retarget.configuration.update_from_keyframe("home")

        initial_t = -1.0
        last_t = initial_t

        # 1. prevent sudden jump of tracked controller position at the begining
        # 2. prevent too aggressive movement of the robot
        x_touch_range = np.array(ik_config["safety_constraints"]["x_touch_range"])
        y_touch_range = np.array(ik_config["safety_constraints"]["y_touch_range"])

        initial_left_ee_pos = np.array(
            ik_config["safety_constraints"]["left_initial_pos"]
        )
        initial_right_ee_pos = np.array(
            ik_config["safety_constraints"]["right_initial_pos"]
        )

        last_left_ee_pos = np.array(ik_config["safety_constraints"]["left_initial_pos"])
        last_right_ee_pos = np.array(
            ik_config["safety_constraints"]["right_initial_pos"]
        )

        delta_pos = np.array(ik_config["safety_constraints"]["delta_pos"])

        initial_pose = {}
        initial_rot = {}
        safety_constrainted_link_pose = {}
        safety_constrainted_link_rot = {}
        # get initial position for anchor joints

        for robot_link, ik_data in ik_match_table.items():
            if robot_link in anchor_list:
                mid = model.body(ik_data[0]).mocapid[0]
                initial_pose[robot_link] = (
                    retarget.configuration.get_transform_frame_to_world(
                        robot_link, ik_data[1]
                    ).translation()
                )
                wxyz = (
                    retarget.configuration.get_transform_frame_to_world(
                        robot_link, ik_data[1]
                    )
                    .rotation()
                    .wxyz
                )
                data.mocap_pos[mid] = initial_pose[robot_link]
                data.mocap_quat[mid] = wxyz
                xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
                # initial_rot[robot_link] = xyzw
                initial_rot[robot_link] = wxyz
            elif robot_link in safety_constrainted_link_list:
                safety_constrainted_link_pose[robot_link] = (
                    retarget.configuration.get_transform_frame_to_world(
                        ik_data[0], "body"
                    ).translation()
                )
                wxyz = (
                    retarget.configuration.get_transform_frame_to_world(
                        ik_data[0], "body"
                    )
                    .rotation()
                    .wxyz
                )
                safety_constrainted_link_rot[robot_link] = wxyz
        # get initial head pose, as the head is in world origin of meta quest
        # head_mid = model.body("head").mocapid[0]
        initial_head_pose = retarget.configuration.get_transform_frame_to_world(
            "head", "body"
        ).translation()
        wxyz = (
            retarget.configuration.get_transform_frame_to_world("head", "body")
            .rotation()
            .wxyz
        )
        # initial_head_rot = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

        # convert the initial rot to rotation matrix
        initial_head_rot = R.from_quat(wxyz, scalar_first=True).as_matrix()

        gripper_target_mid = model.body("left_hand_pose").mocapid[0]

        current_draw_time = time.time()
        last_draw_time = current_draw_time - 0.1
        current_print_time = time.time()
        last_print_time = current_print_time - 0.1

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()
            # get toddy's initial orientation

            # receive quest robot data
            try:
                raw_string = quest.receive()
                (
                    left_hand_pose,
                    left_hand_orn,
                    right_hand_pose,
                    right_hand_orn,
                    head_pose,
                    head_orn,
                ) = quest.string2pos(raw_string, quest.header)

                head_orn_raw = head_orn
                # rotate the camera/world frame to the robot frame in unity configuration (left-handed)
                # robot_head_orn_raw = R.from_quat(head_orn).as_matrix() @ R_z(-90) @ R_x(-90)
                global_rot = R_y(-90)

                left_hand_pose = global_rot.T @ left_hand_pose
                right_hand_pose = global_rot.T @ right_hand_pose
                head_pose = global_rot.T @ head_pose

                head_pose, head_orn = trans_unity_2_robot(
                    head_pose, head_orn, is_quat=True
                )

                left_hand_pose, left_hand_orn = trans_unity_2_robot(
                    left_hand_pose, left_hand_orn, is_quat=True
                )
                right_hand_pose, right_hand_orn = trans_unity_2_robot(
                    right_hand_pose, right_hand_orn, is_quat=True
                )

                # current_print_time = time.time()
                # if current_print_time - last_print_time > 0.1:
                #     last_print_time = current_print_time
                #     print("head pose:", head_pose)

                head_rot_mapping = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
                head_orn = (
                    head_rot_mapping
                    @ R.from_quat(head_orn).as_matrix()
                    @ head_rot_mapping.T
                )
                head_orn = head_orn @ R_y(90) @ R_x(90)
                head_orn = R.from_matrix(head_orn).as_quat()

                lh_rot_mapping = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                left_hand_orn = (
                    lh_rot_mapping
                    @ R.from_quat(left_hand_orn).as_matrix()
                    @ lh_rot_mapping.T
                )
                left_hand_orn = left_hand_orn @ R_z(-90)
                left_hand_orn = R.from_matrix(left_hand_orn).as_quat()

                rh_rot_mapping = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                right_hand_orn = (
                    rh_rot_mapping
                    @ R.from_quat(right_hand_orn).as_matrix()
                    @ rh_rot_mapping.T
                )
                right_hand_orn = right_hand_orn @ R_z(-90) @ R_x(180)
                right_hand_orn = R.from_matrix(right_hand_orn).as_quat()

            except socket.error as e:
                print(e)
                pass
            except KeyboardInterrupt:
                quest.close()
                break

            current_t = time.time()
            if current_t - initial_t < 1.0:
                # prevent unstable control at the beginning
                last_t = current_t
                continue

            if current_t - last_t > 1.0:
                # restart detect
                last_t = current_t
                initial_t = current_t
                continue

            last_t = current_t
            world_origin_pose = np.array(
                [0, 0, 0]
            )  # quest use the character ground as the origin, I guess the height is configured in the hardware
            world_origin_rot = np.eye(3)  # R_z(-90)
            # transform the data in quest frame to mujoco frame
            W_pos_lh = world_origin_rot @ left_hand_pose + world_origin_pose
            W_pos_rh = world_origin_rot @ right_hand_pose + world_origin_pose
            W_pos_head = world_origin_rot @ head_pose + world_origin_pose
            W_rot_lh = world_origin_rot @ R.from_quat(left_hand_orn).as_matrix()
            W_rot_rh = world_origin_rot @ R.from_quat(right_hand_orn).as_matrix()
            W_rot_head = world_origin_rot @ R.from_quat(head_orn).as_matrix()
            # scaling
            W_pos_lh = ik_config["scales"]["left_ee_center"] * W_pos_lh
            W_pos_rh = ik_config["scales"]["right_ee_center"] * W_pos_rh
            W_pos_head = ik_config["scales"]["head"] * W_pos_head

            W_pos_offset = np.array([-W_pos_head[0], -W_pos_head[1], 0])

            W_pos_lh += W_pos_offset
            W_pos_rh += W_pos_offset
            W_pos_head += W_pos_offset

            # W_pos_lh = np.clip(
            #     W_pos_lh, last_left_ee_pos - delta_pos, last_left_ee_pos + delta_pos
            # )
            # W_pos_rh = np.clip(
            #     W_pos_rh, last_right_ee_pos - delta_pos, last_right_ee_pos + delta_pos
            # )

            if is_touch(W_pos_lh, x_touch_range, y_touch_range):
                W_pos_lh[0] = initial_left_ee_pos[0]
                W_pos_lh[1] = initial_left_ee_pos[1]
            if is_touch(W_pos_rh, x_touch_range, y_touch_range):
                W_pos_rh[0] = initial_right_ee_pos[0]
                W_pos_rh[1] = initial_right_ee_pos[1]

            # convert the rotation matrix to quaternion
            W_rot_lh = R.from_matrix(W_rot_lh).as_quat(scalar_first=True)
            W_rot_rh = R.from_matrix(W_rot_rh).as_quat(scalar_first=True)
            W_rot_head = R.from_matrix(W_rot_head).as_quat(scalar_first=True)

            last_left_ee_pos = W_pos_lh
            last_right_ee_pos = W_pos_rh

            quest_poses = {
                "left_ee_center": [W_pos_lh, W_rot_lh],
                "right_ee_center": [W_pos_rh, W_rot_rh],
                "head": [W_pos_head, W_rot_head],
            }
            # transform the poses from quest frame to mujoco frame
            vicon_data = {}
            # feed data from the quest to vicon data
            for robot_link, ik_data in ik_match_table.items():
                # robot_link of mujoco, ik_data[0] from vicon
                if robot_link in anchor_list:
                    vicon_data[ik_data[0]] = [
                        initial_pose[robot_link],
                        initial_rot[robot_link],
                    ]
                elif robot_link in quest_poses.keys():
                    vicon_data[ik_data[0]] = [
                        quest_poses[robot_link][0],
                        quest_poses[robot_link][1],
                    ]
                elif robot_link in safety_constrainted_link_list:
                    vicon_data[ik_data[0]] = [
                        safety_constrainted_link_pose[robot_link],
                        safety_constrainted_link_rot[robot_link],
                    ]

            # Draw the task targets for reference
            if args.view_frame:
                current_draw_time = time.time()
                poses = []
                rots = []
                sizes = []
                orientaiton_corrections = []
                for robot_link, ik_data in ik_match_table.items():
                    if ik_data[0] not in vicon_data:
                        continue
                    elif robot_link in ["head", "left_ee_center", "right_ee_center"]:
                        poses.append(
                            ik_config["scale"] * vicon_data[ik_data[0]][0]
                            - retarget.ground
                        )
                        rots.append(
                            R.from_quat(
                                vicon_data[ik_data[0]][1], scalar_first=True
                            ).as_matrix()
                        )
                        sizes.append(0.1)
                        orientaiton_corrections.append(
                            R.from_quat(ik_data[-1], scalar_first=True)
                        )
                if current_draw_time - last_draw_time > 0.1:
                    last_draw_time = current_draw_time
                    draw_frame_batch(
                        poses, rots, viewer, sizes, orientaiton_corrections
                    )

            # Retarget and pose the model
            retarget.update_targets(vicon_data)
            # data.qpos[:] = retarget.retarget(vicon_data).copy()
            data.qpos[:] = retarget.retarget_v2(
                vicon_data, dt=rate.dt, max_iters=10
            ).copy()
            mj.mj_forward(model, data)
            # retarget.configuration.data = data
            # Visualize at fixed FPS.
            viewer.sync()
            t += dt
            rate.sleep()
