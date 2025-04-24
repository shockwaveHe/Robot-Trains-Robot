import mujoco
import mujoco.viewer
import time
import numpy as np
from toddlerbot.manipulation.teleoperation.data_processing import toddy_quest_module
from toddlerbot.manipulation.teleoperation.data_processing.ip_config import *
from toddlerbot.manipulation.teleoperation.data_processing.rigidbodySento import create_primitive_shape
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib import mink_retarget
from toddlerbot.manipulation.teleoperation.data_processing.retarget_lib.src.retarget_lib.utils.draw import draw_frame
from scipy.spatial.transform import Rotation as R
import pybullet as pb
import json
import socket

if __name__ == "__main__":
    # create a mujoco word loading the toddlerbot model
    
    quest = toddy_quest_module.ToddyQuestBimanualModule(VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT)
    
    start_time = time.time()
    fps_counter = 0
    packet_counter = 0
    print("Initialization completed")
    current_ts = time.time()
    frequency = 30
    while True:
        now = time.time()

        if now - current_ts < 1 / frequency:
            continue
        else:
            current_ts = now
        try:
            raw_string = quest.receive()
            left_hand_pos, left_hand_orn, right_hand_pos, right_hand_orn, head_pos, head_orn = quest.string2pos(raw_string, quest.header)
            print("Right hand pose:", right_hand_pos)
        except socket.error as e:
            print(e)
            pass
        except KeyboardInterrupt:
            quest.close()
            break
        else:
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0