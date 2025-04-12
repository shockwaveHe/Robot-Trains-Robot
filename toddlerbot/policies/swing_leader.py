import time
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.finetuning.utils import Timer
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.keyboard import Keyboard
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
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
        # self.keyboard = Keyboard()
        self.ft_sensor = NetFTSensor()
        self.control_dt = 0.02
        self.total_steps = 0

        self.control_dt = 0.02
        self.total_steps = 0
        self.timer = Timer()
        self.timer.start()
        self.init_speed = 0.05
        self.warmup_time = 10.0
        self.speed_period = 60.0
        self.walk_speed_range = [0.2, 0.2]
        self.paused = False
        self.healthy_force_x = [-10.0, 10.0]
        self.healthy_force_y = [-3.0, 3.0]

    def close(self):
        self.zmq_sender.close()
        self.zmq_receiver.close()

    def is_done(self, obs: Obs):
        # TODO: handle done situation
        # return obs.ee_force[0] < self.healthy_force_x[0] or obs.ee_force[0] > self.healthy_force_x[1] \
        #     or obs.ee_force[1] < self.healthy_force_y[0] or obs.ee_force[1] > self.healthy_force_y[1]
        return False

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        # keyboard_inputs = self.keyboard.get_keyboard_input()
        # self.stopped = keyboard_inputs["stop"]
        # if self.paused and keyboard_inputs["resume"]:
        #     print("Resuming the system")
        #     self.paused = False
        # if not self.paused and keyboard_inputs["pause"]:
        #     print("Pausing the system")
        #     self.paused = True
        # self.keyboard.reset()

        action = self.default_motor_pos.copy()
        ee_force, ee_torque = self.ft_sensor.get_smoothed_data()
        control_inputs = {}
        # print('lin vel', lin_vel, 'speed', self.speed, 'arm_vel', obs.arm_ee_vel)
        obs.ee_force = ee_force
        obs.ee_torque = ee_torque
        is_done = self.is_done(obs)
        msg = ZMQMessage(
            time=time.time(),
            arm_force=ee_force,
            arm_torque=ee_torque,
            arm_ee_pos=np.zeros(3, dtype=np.float32),
            arm_ee_vel=np.zeros(3, dtype=np.float32),
            is_stopped=is_done,  # or self.stopped,
            is_paused=self.paused,
            external_guidance_stage="free",
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
            self.total_steps = 0
            self.timer.start()

        time_elapsed = self.timer.elapsed()
        if time_elapsed < self.total_steps * self.control_dt:
            time.sleep(self.total_steps * self.control_dt - time_elapsed)
        self.total_steps += 1

        return control_inputs, action, obs

    def reset(self, obs: Obs = None) -> Obs:
        self.timer.reset()
        # control_inputs = {}
        # msg = ZMQMessage(
        #     time=time.time(),
        #     control_inputs=control_inputs,
        #     is_done=True
        # )
        # self.zmq_sender.send_msg(msg)
        print("Reset done")

        self.timer.start()
        return obs
