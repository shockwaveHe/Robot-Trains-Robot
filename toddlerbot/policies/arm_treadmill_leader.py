import struct
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial

from toddlerbot.finetuning.utils import Timer
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
        eval_mode: bool = False,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.zmq_sender = ZMQNode(type="sender", ip=ip)
        self.zmq_receiver = ZMQNode(type="receiver")

        print(f"ZMQ Connected to {ip}")

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False
        self.eval_mode = eval_mode
        self.enable_keyboard = False

        if keyboard is None:
            self.keyboard = Keyboard()
        else:
            self.keyboard = keyboard

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.speed = 0.0
        self.speed_wt_peturbation = 0.0
        self.max_speed = 240.0
        self.min_speed = 0.0
        self.base_speed = 100.0

        self.walk_x = 0.0
        self.walk_y = 0.0

        self.stopped = False
        self.force = 20.0
        self.z_pos_delta = 0.0

        self.serial_thread = threading.Thread(target=self.serial_thread_func)
        self.serial_thread.start()

        shm_name = "force_shm"
        try:
            print("Creating shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=112
            )
        except FileExistsError:
            print("Using existing shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=False, size=112
            )
        self.arm_shm.buf[:8] = struct.pack("d", self.force)
        self.arm_shm.buf[8:16] = struct.pack("d", self.z_pos_delta)

        # TODO: put this logic and reset to realworld finetuning sim?
        self.x_force_threshold = 0.5
        self.treadmill_speed_inc_kp = 1.5
        self.treadmill_speed_dec_kp = 1.0
        self.treadmill_speed_force_kp = 20.0
        self.treadmill_speed_pos_kp = -20.0
        self.ee_force_x_ema = 0.0
        self.ee_force_x_ema_alpha = 0.2
        self.arm_healthy_ee_pos_y = np.array([-0.3, 0.3])
        self.arm_healthy_ee_force_z = np.array([-10.0, 40.0])
        self.arm_healthy_ee_force_xy = np.array([-10.0, 10.0])
        self.healthy_torso_roll = np.array([-0.3, 0.3])
        self.healthy_torso_pitch = np.array([-0.3, 0.3])

        self.stream_started = False
        self.reset_counter = 0
        self.reset_count_max = 200

        self.use_torso_pitch_feedback = True
        self.current_torso_roll = 0.0
        self.current_torso_pitch = 0.0
        self.torso_pitch_dec_kp = 500.0
        self.torso_pitch_inc_kp = 500.0
        # self.speed_stall_window = 3
        # self.speed_delta_buffer = deque(maxlen=self.speed_stall_window)
        # for _ in range(self.speed_stall_window):
        #     self.speed_delta_buffer.append(np.random.rand())

        self.init_arm_pos = init_arm_pos.copy()
        print("init arm pos", self.init_arm_pos)
        self.timer = Timer()
        self.timer.start()
        self.init_speed = 0.05
        self.warmup_time = 10.0
        self.speed_period = 60.0
        self.walk_speed_range = [0.2, 0.2]

        self.schedule_types = [
            "arm_stepwise",
            # "treadmill_stepwise",
            "arm_eval",
            "treadmill_peturbation_stepwise",
            "arm_treadmill_stepwise",
            # "arm_feedback_gain_stepwise",
        ]
        self.schedule_type = "arm_stepwise"
        # arm schedule
        self.total_train_steps = 50000
        self.total_evaluation_steps = 6000

        # Different schedules 

        # if self.schedule_type in [
        #     "arm_stepwise",
        #     "arm_treadmill_stepwise",
        #     "arm_feedback_gain_stepwise",
        # ]:
        #     self.arm_train_schedule = (
        #         np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * self.total_train_steps
        #     )
        #     self.train_z_pos_target_schedule = np.array(
        #         [-0.01, -0.015, -0.02, -0.025, -0.03]
        #     )

        #     assert len(self.arm_train_schedule) == len(
        #         self.train_z_pos_target_schedule
        #     ), "train schedule and z pos target schedule must have the same length"
        #     self.arm_schedule_mapping = {
        #         self.arm_train_schedule[i]: self.train_z_pos_target_schedule[i]
        #         for i in range(len(self.arm_train_schedule))
        #     }

        # if self.schedule_type in ["treadmill_stepwise", "arm_treadmill_stepwise"]:
        #     self.treadmill_train_schedule = (
        #         np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * self.total_train_steps
        #     )
        #     self.target_speed_weight = np.array([0.00, 0.25, 0.5, 0.75, 1.0])
        #     self.target_speed_weight = np.exp((self.target_speed_weight - 1) * 2.5)
        #     self.target_speed_weight[0] = 0.0
        #     self.treadmill_schedule_mapping = {
        #         self.treadmill_train_schedule[i]: self.target_speed_weight[i]
        #         for i in range(len(self.treadmill_train_schedule))
        #     }

        if self.schedule_type in [
            "treadmill_peturbation_stepwise",
            "arm_treadmill_stepwise",
        ]:
            self.treadmill_train_schedule = (
                np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * self.total_train_steps
            )
            # Gaussian noise added to the actual treadmill speed
            self.perturbation_noise_level = 20
            self.peturbation_noise_schedule = (
                np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * self.perturbation_noise_level
            )

            self.peturbation_noise_schedule_mapping = {
                self.treadmill_train_schedule[i]: self.peturbation_noise_schedule[i]
                for i in range(len(self.treadmill_train_schedule))
            }

        if self.schedule_type == "arm_eval":
            print("Using default arm evaluation schedule")

        if self.schedule_type == "arm_feedback_gain_stepwise":
            self.arm_feedback_gain_train_schedule = (
                np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * self.total_train_steps
            )
            self.treadmill_speed_force_kp_schedule = (
                np.array([1.0, 1.25, 1.5, 1.75, 2.0]) * self.treadmill_speed_force_kp
            )
            self.treadmill_speed_force_kp_schedule_mapping = {
                self.arm_feedback_gain_train_schedule[
                    i
                ]: self.treadmill_speed_force_kp_schedule[i]
                for i in range(len(self.arm_feedback_gain_train_schedule))
            }

        self.current_train_steps = 0
        self.evaluation_begin = False
        self.test_begin = False
        self.evaluation_start_time = 0.0
        self.init_z_pos_target = -0.01
        self.evaluate_z_pos_target = -0.03
        self.treadmill_eval_speed = 150.0
        self.arm_speed = -0.2  # m/s
        self.paused = False
        self.treadmill_pause_time = time.time()

    def close(self):
        self.zmq_sender.close()
        self.serial_thread.join()
        self.arm_shm.close()
        try:
            self.arm_shm.unlink()
        except FileNotFoundError:
            pass

    def update_speed(self, obs: Obs, total_steps=0):
        if time.time() - self.treadmill_pause_time < 1.0:
            return

        if total_steps >= self.total_train_steps + self.total_evaluation_steps:
            self.test_begin = True
            # print("Test begin")
        elif total_steps >= self.total_train_steps:
            self.evaluation_begin = True
            # print("Evaluation begin")
        if not self.stream_started:
            self.speed = 0.0
            self.speed_wt_peturbation = 0.0
            print("Waiting for the follower to start...")
            return

        if self.test_begin:
            self.speed = self.treadmill_eval_speed
            self.speed_wt_peturbation = self.treadmill_eval_speed
            print(
                f"set treadmill speed to evaluation speed {self.treadmill_eval_speed / 1000} m/s"
            )
            return

        self.speed = self.base_speed
        self.speed += self.treadmill_speed_force_kp * obs.ee_force[0]
        if self.use_torso_pitch_feedback:
            # if (
            #     self.current_torso_pitch < self.healthy_torso_pitch[0]
            #     or self.current_torso_pitch > self.healthy_torso_pitch[1]
            # ):
            if self.current_torso_pitch > 0:
                self.speed -= self.torso_pitch_dec_kp * self.current_torso_pitch
            else:
                self.speed -= self.torso_pitch_inc_kp * self.current_torso_pitch

        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

        if self.schedule_type in [
            "arm_treadmill_stepwise",
            "treadmill_peturbation_stepwise",
        ]:
            additive_noise = self.treadmill_perturbation_schedule(
                self.current_train_steps
            )
            print("additive noise:", additive_noise)
            self.speed_wt_peturbation = self.speed + additive_noise
            self.speed_wt_peturbation = np.clip(
                self.speed_wt_peturbation, 0.0, self.max_speed
            )

        print(
            f"Force X, Y: {obs.ee_force[0]:.4f}, {obs.ee_force[1]:.4f} treadmill speed: {self.speed:.4f}"
        )
        # print(f"Delta Arm pose x: {delta_arm_pos_x:.3f}, Arm pose: {obs.arm_ee_pos[0]:.3f} {obs.arm_ee_pos[1]:.3f}, Arm vel: {obs.arm_ee_vel[0]:.3g} {obs.arm_ee_vel[1]:.3g}, Force: {obs.ee_force[0]:.3f} {obs.ee_force[1]:.3f} {obs.ee_force[2]:.3f}")

    # speed not enough is x negative
    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def treadmill_perturbation_schedule(self, total_steps):
        # if total_steps < self.total_train_steps:
        #     return 0.0

        for key, value in self.peturbation_noise_schedule_mapping.items():
            if total_steps < key:
                additive_noise = np.random.normal(loc=0.0, scale=value, size=(1,))
                break
        if total_steps >= self.total_train_steps + self.total_evaluation_steps:
            additive_noise = 0.0
            self.test_begin = True
            print("Test begin")
        elif total_steps >= self.total_train_steps:
            additive_noise = 0.0
            self.evaluation_begin = True
            print("Evaluation begin")

        return float(additive_noise)

    def arm_feedback_gain_schedule(self, total_steps):
        treadmill_speed_force_kp = self.treadmill_speed_force_kp
        for key, value in self.treadmill_speed_force_kp_schedule_mapping.items():
            if total_steps < key:
                treadmill_speed_force_kp = value
                break
        if total_steps >= self.total_train_steps:
            treadmill_speed_force_kp = self.treadmill_speed_force_kp
        return treadmill_speed_force_kp

    def arm_height_default_schedule(self, total_steps):
        if total_steps < self.total_train_steps:
            return self.evaluate_z_pos_target
        if not self.evaluation_begin:
            self.evaluation_begin = True
            self.evaluation_start_time = time.time()
            print("Evaluation begin")
        # evaluation_time = time.time() - self.evaluation_start_time
        # delta_z_target = self.arm_speed * evaluation_time
        # delta_z_target = np.max([delta_z_target, self.evaluate_z_pos_target])
        # negative
        return self.evaluate_z_pos_target

    def arm_height_stepwise_schedule(self, total_steps):
        # delta_z_target = 0
        # for key, value in self.arm_schedule_mapping.items():
        #     if total_steps < key:
        #         delta_z_target = value
        #         break
        delta_z_target = self.init_z_pos_target + (
            total_steps
            / self.total_train_steps
            * (self.evaluate_z_pos_target - self.init_z_pos_target)
        )
        if total_steps >= self.total_train_steps:
            delta_z_target = self.evaluate_z_pos_target
            self.evaluation_begin = True
        return delta_z_target

    def walk_speed_schedule(self):
        current_time = self.timer.elapsed()
        min_speed, max_speed = self.walk_speed_range

        if current_time < self.warmup_time:
            # Linear interpolation from init_speed to min_speed
            speed = self.init_speed + (min_speed - self.init_speed) * (
                current_time / self.warmup_time
            )
        else:
            # Sinusoidal oscillation between min_speed and max_speed
            phase = (current_time - self.warmup_time) / self.speed_period * 2 * np.pi
            speed = min_speed + (max_speed - min_speed) * (0.5 * (1 - np.cos(phase)))

        return speed

    def is_done(self, obs: Obs):
        if (
            obs.ee_force[2] < self.arm_healthy_ee_force_z[0]
            or obs.ee_force[2] > self.arm_healthy_ee_force_z[1]
        ):
            print(f"Force Z of {obs.ee_force[2]} is out of range")
            self.reset_counter += 1
        elif (
            obs.ee_force[0] > self.arm_healthy_ee_force_xy[1]
            or obs.ee_force[1] > self.arm_healthy_ee_force_xy[1]
        ):
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            self.reset_counter += 1
        elif (
            obs.ee_force[0] < self.arm_healthy_ee_force_xy[0]
            or obs.ee_force[1] < self.arm_healthy_ee_force_xy[0]
        ):
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            self.reset_counter += 1
        elif (
            self.current_torso_roll < self.healthy_torso_roll[0]
            or self.current_torso_roll > self.healthy_torso_roll[1]
        ):
            print(f"Torso Roll of {self.current_torso_roll} is out of range")
            self.reset_counter += 1
        elif (
            self.current_torso_pitch < self.healthy_torso_pitch[0]
            or self.current_torso_pitch > self.healthy_torso_pitch[1]
        ):
            print(f"Torso Pitch of {self.current_torso_pitch} is out of range")
            self.reset_counter += 1
        else:
            self.reset_counter = 0

        if self.reset_counter > self.reset_count_max:
            self.reset_counter = 0
            return True
        else:
            return False

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        self.walk_x = self.walk_speed_schedule()
        keyboard_inputs = self.keyboard.get_keyboard_input()
        if self.enable_keyboard:
            self.walk_x += keyboard_inputs["walk_x_delta"]
            self.walk_y += keyboard_inputs["walk_y_delta"]

        control_inputs = {
            "walk_x": self.walk_x,
            "walk_y": self.walk_y,
            "walk_turn": 0.0,
        }

        self.stopped = keyboard_inputs["stop"]
        if self.enable_keyboard and self.paused and keyboard_inputs["resume"]:
            print("Resuming the system")
            self.paused = False
        if self.enable_keyboard and not self.paused and keyboard_inputs["pause"]:
            print("Pausing the system")
            self.paused = True
        if self.eval_mode:
            self.speed = self.walk_x * 1000
        else:
            self.update_speed(obs, self.current_train_steps)
        if self.enable_keyboard:
            self.speed += keyboard_inputs["speed_delta"]
            self.force += keyboard_inputs["force_delta"]
        self.arm_shm.buf[:8] = struct.pack("d", self.force)
        if self.enable_keyboard:
            self.z_pos_delta += keyboard_inputs["z_pos_delta"]  # incremental
        elif self.schedule_type in [
            "arm_stepwise",
            "arm_treadmill_stepwise",
            "arm_feedback_gain_stepwise",
        ]:
            self.z_pos_delta = self.arm_height_stepwise_schedule(
                self.current_train_steps
            )
        else:
            self.z_pos_delta = self.arm_height_default_schedule(
                self.current_train_steps
            )

        print(f"z_pos_delta {self.z_pos_delta}")
        # if self.enable_keyboard:
        #     self.arm_shm.buf[8:16] = struct.pack(
        #         "d", keyboard_inputs["z_pos_delta"]
        #     )  # not add equal
        # else:  # remember to change the c++ code, now the delta refers to difference from the initial z pos
        self.arm_shm.buf[8:16] = struct.pack("d", self.z_pos_delta)
        self.keyboard.reset()
        # print(f"force {self.force}, speed {self.speed}, z_pos_delta {self.z_pos_delta}", control_inputs)

        action = self.default_motor_pos.copy()

        if self.stopped:
            self.force = 0.0
            self.z_pos_delta = 0.0
            self.arm_shm.buf[:8] = struct.pack("d", self.force)
            self.arm_shm.buf[8:16] = struct.pack("d", self.z_pos_delta)
            self.keyboard.close()
            control_inputs = {"walk_x": 0.0, "walk_y": 0.0, "walk_turn": 0.0}
            print("Stopping the system")

        if self.paused:
            self.speed = 0.0
            self.walk_x = 0.0
            self.walk_y = 0.0
            # How to stop at current pose
            control_inputs = {"walk_x": 0.0, "walk_y": 0.0, "walk_turn": 0.0}

        # compile data to send to follower
        assert control_inputs is not None
        lin_vel = obs.arm_ee_vel.copy()
        lin_vel[0] = self.speed / 1000  # - lin_vel[0]
        lin_vel[1] = -lin_vel[1]
        # print('lin vel', lin_vel, 'speed', self.speed, 'arm_vel', obs.arm_ee_vel)
        is_done = self.is_done(obs)
        self.paused = is_done
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            arm_force=obs.ee_force,
            arm_torque=obs.ee_torque,
            arm_ee_pos=obs.arm_ee_pos,
            lin_vel=lin_vel,  # TODO: check if this is correct
            is_done=is_done,
            is_stopped=self.stopped,
            is_paused=self.paused,
        )
        # import ipdb; ipdb.set_trace()
        if not self.paused:
            print(
                f"Speed: {self.speed}, Speed pb: {self.speed_wt_peturbation}, Walk: ({self.walk_x}, {self.walk_y}), Arm Delta: {self.z_pos_delta}"
            )
        self.zmq_sender.send_msg(msg)
        msg_recv = self.zmq_receiver.get_msg()
        if msg_recv is not None:
            if not self.stream_started:
                self.stream_started = True
                print("Stream started")
            if msg_recv.total_steps > 0:
                self.current_train_steps = msg_recv.total_steps
            if msg_recv.torso_pitch is not None:
                self.current_torso_pitch = msg_recv.torso_pitch
            if msg_recv.torso_roll is not None:
                self.current_torso_roll = msg_recv.torso_roll
                # print(f"Received train steps: {self.current_train_steps}")
        if msg_recv is not None and msg_recv.is_stopped:
            self.timer.reset()
            self.speed = 0.0
            self.speed_wt_peturbation = 0.0
            self.walk_x = 0.0
            self.walk_y = 0.0
            print("Waiting for the follower to start...")
            force_prev = self.force
            self.arm_shm.buf[:8] = struct.pack("d", -2.0)
            while True:
                msg_recv = self.zmq_receiver.get_msg()
                if msg_recv is not None and not msg_recv.is_stopped:
                    break
                time.sleep(0.1)
            print("Follower started")
            self.walk_x = 0.05
            self.arm_shm.buf[:8] = struct.pack("d", force_prev)
            self.timer.start()

        if self.stopped:
            self.serial_thread.join()
            self.arm_shm.close()
            self.arm_shm.unlink()
        return control_inputs, action, obs

    def serial_thread_func(self):
        # Configure the serial connection
        ser = serial.Serial(
            port="/dev/ttyUSB0",  # Replace with your port name on Linux
            baudrate=9600,  # Baud rate
            parity=serial.PARITY_NONE,  # Parity (None)
            stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            bytesize=serial.EIGHTBITS,  # Data bits (8)
            timeout=0.1,  # Set a timeout to prevent blocking
        )

        # Check if the port is open
        if ser.is_open:
            print(f"Connected to {ser.name}")

        try:
            while not self.stopped:
                # Get the current speed
                if self.schedule_type in [
                    "treadmill_peturbation_stepwise",
                    "arm_treadmill_stepwise",
                ]:
                    current_speed = self.speed_wt_peturbation
                else:
                    current_speed = self.speed

                # Prepare the data to send
                data_to_send = f"{current_speed}\n".encode("utf-8")
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
        self.timer.reset()
        lin_vel = np.zeros(3)
        lin_vel[0] = self.speed / 1000
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            arm_force=np.zeros(3),
            arm_torque=np.zeros(3),
            arm_ee_pos=np.zeros(3),
            lin_vel=lin_vel,
            is_done=True,
        )
        self.zmq_sender.send_msg(msg)
        self.speed = 0.0
        self.speed_wt_peturbation = 0.0
        self.walk_x = 0.0
        self.walk_y = 0.0
        self.current_torso_roll = 0.0
        self.current_torso_pitch = 0.0
        force_prev = self.force
        self.force = -0.5
        self.arm_shm.buf[:8] = struct.pack("d", self.force)
        # input("Press Enter to reset...")
        self.force = -1.0
        self.arm_shm.buf[:8] = struct.pack("d", self.force)

        cur_force = struct.unpack("d", self.arm_shm.buf[:8])[0]
        print(f"Waiting for force to reset... {cur_force}")
        while cur_force == -1.0:
            self.zmq_sender.send_msg(msg)
            cur_force = struct.unpack("d", self.arm_shm.buf[:8])[0]
            time.sleep(0.2)
        assert cur_force == force_prev
        print("Reset done")
        self.treadmill_pause_time = time.time()
        self.force = force_prev
        self.timer.start()
        return obs

    def reset_after(self, duration: float):
        self.reset_time = time.time() + duration
