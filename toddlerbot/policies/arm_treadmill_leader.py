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

class ArmTreadmillLeaderPolicy(BasePolicy, policy_name="at_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        init_arm_pos: npt.NDArray[np.float32],
        keyboard: Optional[Keyboard] = None,
        ip: str = "192.168.0.70",
    ):
        super().__init__(name, robot, init_motor_pos)

        self.zmq_sender = ZMQNode(type="sender", ip=ip)
        self.zmq_receiver = ZMQNode(type="receiver")

        print(f"ZMQ Connected to {ip}")

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        if keyboard is None:
            self.keyboard = Keyboard()
        else:
            self.keyboard = keyboard

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.speed = 0.0
        self.max_speed = 240.0

        self.walk_x = 0.05
        self.walk_y = 0.0

        self.stopped = False
        self.force = 20.0
        self.z_pos_delta = 0.0

        self.serial_thread = threading.Thread(target=self.serial_thread_func)
        self.serial_thread.start()

        shm_name = 'force_shm'
        try:
            print("Creating shared memory")
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=112)
        except FileExistsError:
            print("Using existing shared memory")
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=False, size=112)
        self.arm_shm.buf[:8] = struct.pack('d', self.force)
        self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)

        # TODO: put this logic and reset to realworld finetuning sim?
        self.x_force_threshold = 0.5
        self.treadmill_speed_inc_kp = 2.5
        self.treadmill_speed_dec_kp = 2.5
        self.treadmill_speed_force_kp = 1.0
        self.treadmill_speed_pos_kp = -1.0
        self.ee_force_x_ema = 0.0
        self.ee_force_x_ema_alpha = 0.2
        self.arm_healthy_ee_pos = np.array([0.0, 3.0])
        self.arm_healthy_ee_force_z = np.array([-10.0, 40.0])
        self.arm_healthy_ee_force_xy = np.array([-5.0, 5.0])
        self.healthy_torso_roll = np.array([-0.5, 0.5])
        self.healthy_torso_pitch = np.array([-0.5, 0.5])
        self.speed_stall_window = 3
        self.speed_delta_buffer = deque(maxlen=self.speed_stall_window)
        for _ in range(self.speed_stall_window):
            self.speed_delta_buffer.append(np.random.rand())

        self.init_arm_pos = init_arm_pos.copy()
        print('init arm pos', self.init_arm_pos)

    def close(self):
        self.zmq_sender.close()
        self.serial_thread.join()
        self.arm_shm.close()
        try:
            self.arm_shm.unlink()
        except FileNotFoundError:
            pass
        
    def update_speed(self, obs: Obs):
        self.ee_force_x_ema = self.ee_force_x_ema_alpha * self.ee_force_x_ema + (1 - self.ee_force_x_ema_alpha) * obs.ee_force[0]
        delta_speed = np.abs(self.ee_force_x_ema) - self.x_force_threshold
        speed_stalled = True
        for prev_delta_speed in self.speed_delta_buffer:
            if not np.allclose(prev_delta_speed, delta_speed):
                speed_stalled = False
                break
        delta_arm_pos_x = obs.arm_ee_pos[0] - self.init_arm_pos[0]
        self.speed_delta_buffer.append(delta_speed)
        if speed_stalled:
            print(f"Speed delta stalled at {delta_speed}")
            return
        delta_speed = self.treadmill_speed_force_kp * delta_speed + self.treadmill_speed_pos_kp * delta_arm_pos_x
        delta_speed = np.clip(delta_speed, 0.0, 1.0)
        if self.ee_force_x_ema > self.x_force_threshold:
            delta_speed *= self.treadmill_speed_inc_kp
            # print(f"about to increase speed {delta_speed}")
            self.speed += delta_speed
            self.speed = min(self.max_speed, self.speed)
        elif self.ee_force_x_ema < -self.x_force_threshold:
            delta_speed *= self.treadmill_speed_dec_kp
            # print(f"about to decrease speed {delta_speed}")
            self.speed -= delta_speed
            self.speed = max(0.0, self.speed)
        print(f"Delta Arm Pos X: {delta_arm_pos_x}, Force X: {obs.ee_force[0]}, Delta Speed: {delta_speed}")
    # speed not enough is x negative
    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json
    
    def is_done(self, obs: Obs):
        if obs.ee_force[2] < self.arm_healthy_ee_force_z[0] or obs.ee_force[2] > self.arm_healthy_ee_force_z[1]:
            print(f"Force Z of {obs.ee_force[2]} is out of range")
            return True
        if obs.arm_ee_pos[2] < self.arm_healthy_ee_pos[0] or obs.arm_ee_pos[2] > self.arm_healthy_ee_pos[1]:
            print(f"Position Z of {obs.arm_ee_pos[2]} is out of range")
            return True
        if obs.ee_force[0] > self.arm_healthy_ee_force_xy[1] or obs.ee_force[1] > self.arm_healthy_ee_force_xy[1]:
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            return True
        if obs.ee_force[0] < self.arm_healthy_ee_force_xy[0] or obs.ee_force[1] < self.arm_healthy_ee_force_xy[0]:
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            return True
        if obs.euler[0] < self.healthy_torso_roll[0] or obs.euler[0] > self.healthy_torso_roll[1]:
            print(f"Torso Roll of {obs.euler[0]} is out of range")
            return True
        if obs.euler[1] < self.healthy_torso_pitch[0] or obs.euler[1] > self.healthy_torso_pitch[1]:
            print(f"Torso Pitch of {obs.euler[1]} is out of range")
            return True
        return False
    
    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        keyboard_inputs = self.keyboard.get_keyboard_input()
        self.walk_x += keyboard_inputs["walk_x_delta"]
        self.walk_y += keyboard_inputs["walk_y_delta"]
        control_inputs = {"walk_x": self.walk_x, "walk_y": self.walk_y, "walk_turn": 0.0}

        self.stopped = keyboard_inputs["stop"]
        self.update_speed(obs)
        self.speed += keyboard_inputs["speed_delta"]
        self.force += keyboard_inputs["force_delta"]
        self.arm_shm.buf[:8] = struct.pack('d', self.force)
        self.z_pos_delta += keyboard_inputs["z_pos_delta"]
        self.arm_shm.buf[8:16] = struct.pack('d', keyboard_inputs["z_pos_delta"]) # not add equal
        self.keyboard.reset()
        # print(f"force {self.force}, speed {self.speed}, z_pos_delta {self.z_pos_delta}", control_inputs)

        action = self.default_motor_pos.copy()

        if self.stopped:
            self.force = 0.0
            self.z_pos_delta = 0.0
            self.arm_shm.buf[:8] = struct.pack('d', self.force)
            self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
            self.keyboard.close()
            control_inputs = {"walk_x": 0.0, "walk_y": 0.0, "walk_turn": 0.0}
            print("Stopping the system")

        # compile data to send to follower
        assert control_inputs is not None
        lin_vel = obs.arm_ee_vel.copy()
        lin_vel[0] = self.speed / 1000 - lin_vel[0]
        lin_vel[1] = -lin_vel[1]
        # print('lin vel', lin_vel, 'speed', self.speed, 'arm_vel', obs.arm_ee_vel)
        is_done = self.is_done(obs)
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            arm_force=obs.ee_force,
            arm_torque=obs.ee_torque,
            arm_ee_pos=obs.arm_ee_pos,
            lin_vel=lin_vel, # TODO: check if this is correct
            is_done=is_done,
            is_stopped=self.stopped
        )
        # import ipdb; ipdb.set_trace()
        print(f"Speed: {self.speed}, Force: {self.force}, Walk: ({self.walk_x}, {self.walk_y}), Arm Delta: {self.z_pos_delta}")
        self.zmq_sender.send_msg(msg)
        msg_recv = self.zmq_receiver.get_msg()
        if msg_recv is not None and msg_recv.is_stopped:
            msg = ZMQMessage(
                time=time.time(),
                control_inputs=control_inputs,
                arm_force=np.zeros(3),
                arm_torque=np.zeros(3),
                arm_ee_pos=np.zeros(3),
                lin_vel=lin_vel,
                is_done=True
            )
            self.zmq_sender.send_msg(msg)
            self.speed = 0.0
            self.walk_x = 0.0
            self.walk_y = 0.0
            print("Waiting for the follower to start...")
            force_prev = self.force
            self.arm_shm.buf[:8] = struct.pack('d', -2.0)
            while True:
                msg_recv = self.zmq_receiver.get_msg()
                if msg_recv is not None and not msg_recv.is_stopped:
                    break
                time.sleep(0.1)
            print("Follower started")
            self.walk_x = 0.05
            self.arm_shm.buf[:8] = struct.pack('d', force_prev)

        if self.stopped:
            self.serial_thread.join()
            self.arm_shm.close()
            self.arm_shm.unlink()
        return control_inputs, action

    def serial_thread_func(self):
        # Configure the serial connection
        ser = serial.Serial(
            port='/dev/ttyUSB0',     # Replace with your port name on Linux
            baudrate=9600,           # Baud rate
            parity=serial.PARITY_NONE,    # Parity (None)
            stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            bytesize=serial.EIGHTBITS,     # Data bits (8)
            timeout=0.1              # Set a timeout to prevent blocking
        )

        # Check if the port is open
        if ser.is_open:
            print(f"Connected to {ser.name}")

        try:
            while not self.stopped:
                # Get the current speed
                current_speed = self.speed

                # Prepare the data to send
                data_to_send = f"{current_speed}\n".encode('utf-8')
                ser.write(data_to_send)  # Send the data

                # Optionally read a response (if the device sends back data)
                _ = ser.readline()  # Read a line from the device
                # if response:
                #     print(f"Send: {data_to_send}, Received: {response}")

                # Sleep to prevent excessive CPU usage
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in serial thread: {e}")
        finally:
            # Send zero speed before closing
            ser.write(b"0\n")
            ser.close()
            print("Serial connection closed.")

    def reset(self, obs: Obs = None) -> Obs:
        control_inputs = {"walk_x": 0.0, "walk_y": 0.0, "walk_turn": 0.0}
        lin_vel = np.zeros(3)
        lin_vel[0] = self.speed / 1000
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            arm_force=np.zeros(3),
            arm_torque=np.zeros(3),
            arm_ee_pos=np.zeros(3),
            lin_vel=lin_vel,
            is_done=True
        )
        self.zmq_sender.send_msg(msg)
        self.speed = 0.0
        self.walk_x = 0.0
        self.walk_y = 0.0
        input("Press Enter to reset...")
        # TODO: more safe reset
        force_prev = self.force
        self.force = -1.0
        self.arm_shm.buf[:8] = struct.pack('d', self.force)
        # self.force = 30.0
        # self.arm_shm.buf[:8] = struct.pack('d', self.force)
        # for _ in range(10):
        #     self.z_pos_delta = 0.01
        #     self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
        #     time.sleep(0.5)
        # for _ in range(10):
        #     self.z_pos_delta = -0.01
        #     self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
        #     time.sleep(0.5)
        # input("Press Enter to finish...")
        # import ipdb; ipdb.set_trace()
        cur_force = struct.unpack('d', self.arm_shm.buf[:8])[0]
        print(f"Waiting for force to reset... {cur_force}")
        while cur_force == -1.0:
            cur_force = struct.unpack('d', self.arm_shm.buf[:8])[0]
            time.sleep(0.2)
        assert cur_force == force_prev
        print("Reset done")
        self.force = force_prev
        # TODO: how to restart datacollection?
        return obs

    def reset_after(self, duration: float):
        self.reset_time = time.time() + duration