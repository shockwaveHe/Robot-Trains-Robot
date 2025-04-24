import datetime
import os
import shutil
import socket
import time

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from toddlerbot.manipulation.teleoperation.data_processing.quest_robot_module import (
    QuestRobotModule,
)


class ToddyQuestBimanualModule(QuestRobotModule):
    def __init__(self, vr_ip, local_ip, pose_cmd_port, ik_result_port, vis_sp=None):
        super().__init__(vr_ip, local_ip, pose_cmd_port, ik_result_port)
        self.vis_sp = vis_sp
        # check if file exist
        # self.toddy = pb.loadURDF("../../../toddlerbot/descriptions/toddlerbot_active/toddlerbot_active.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0, 1.0], useFixedBase=True)
        self.last_arm_q = None
        self.last_hand_q = None
        self.last_action = 1
        self.last_action_t = time.time()
        self.header = "Bihand and Head: "

    def receive(self):
        data, _ = self.wrist_listener_s.recvfrom(1024)
        data_string = data.decode()
        # remove the header from data_string
        return data_string

    def string2pos(self, data_string, header):
        data_string = data_string[len(header) :]
        data_string = data_string.split(",")
        data_list = [float(i) for i in data_string]
        left_hand_pos = np.array(data_list[:3])
        left_hand_orn = np.array(data_list[3:7])  # x, y, z, w
        right_hand_pos = np.array(data_list[7:10])
        right_hand_orn = np.array(data_list[10:14])
        head_pos = np.array(data_list[14:17])
        head_orn = np.array(data_list[17:21])
        return (
            left_hand_pos,
            left_hand_orn,
            right_hand_pos,
            right_hand_orn,
            head_pos,
            head_orn,
        )
