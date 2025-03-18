from collections import deque
from multiprocessing import shared_memory
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.keyboard import Keyboard
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.finetuning.utils import Timer
from toddlerbot.utils.ft_utils import NetFTSensor

class SwingLeaderPolicy(BasePolicy, policy_name="swing_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "192.168.0.70",
    ):
        super().__init__(name, robot, init_motor_pos)

        self.zmq_sender = ZMQNode(type="sender", ip=ip)
        self.zmq_receiver = ZMQNode(type="receiver")

        print(f"ZMQ Connected to {ip}")

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.ft_sensor = NetFTSensor()
        self.speed_stall_window = 3
        self.speed_delta_buffer = deque(maxlen=self.speed_stall_window)
        for _ in range(self.speed_stall_window):
            self.speed_delta_buffer.append(np.random.rand())

        self.timer = Timer()
        self.timer.start()
        self.init_speed = 0.05
        self.warmup_time = 10.0
        self.speed_period = 60.0
        self.walk_speed_range = [0.2, 0.2]

        self.treadmill_pause_time = time.time()

    def close(self):
        self.zmq_sender.close()
        self.zmq_receiver.close()

    def is_done(self, obs: Obs):
        return False
    
    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:

        action = self.default_motor_pos.copy()
        ee_force, ee_torque = self.ft_sensor.get_smoothed_data()
        control_inputs = {}
        # print('lin vel', lin_vel, 'speed', self.speed, 'arm_vel', obs.arm_ee_vel)
        is_done = self.is_done(obs)
        msg = ZMQMessage(
            time=time.time(),
            arm_force=ee_force,
            arm_torque=ee_torque,
            is_stopped=is_done
        )
        # import ipdb; ipdb.set_trace()
        self.zmq_sender.send_msg(msg)
        msg_recv = self.zmq_receiver.get_msg()
        if msg_recv is not None and msg_recv.is_stopped:
            self.timer.reset()

            print("Waiting for the follower to start...")
            while True:
                msg_recv = self.zmq_receiver.get_msg()
                if msg_recv is not None and not msg_recv.is_stopped:
                    break
                time.sleep(0.1)
            print("Follower started")
            self.timer.start()


        return control_inputs, action


    def reset(self, obs: Obs = None) -> Obs:
        self.timer.reset()
        control_inputs = {}
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            is_done=True
        )
        self.zmq_sender.send_msg(msg)
        print("Reset done")

        self.timer.start()
        return obs