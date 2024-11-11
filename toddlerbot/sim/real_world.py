from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports

# from toddlerbot.utils.misc_utils import profile


class RealWorld(BaseSim):
    def __init__(self, robot: Robot):
        super().__init__("real_world")
        self.robot = robot

        self.has_imu = self.robot.config["general"]["has_imu"]
        self.has_dynamixel = self.robot.config["general"]["has_dynamixel"]
        self.has_sunny_sky = self.robot.config["general"]["has_sunny_sky"]

        # TODO: Fix the mate directions in the URDF and remove the negated_motor_names
        self.negated_motor_names: List[str] = [
            "neck_pitch_act",
            "left_sho_roll",
            "right_sho_roll",
            "left_elbow_roll",
            "right_elbow_roll",
            "left_wrist_pitch_drive",
            "right_wrist_pitch_drive",
            "left_gripper_rack",
            "right_gripper_rack",
        ]

        self.initialize()

    def initialize(self) -> None:
        self.executor = ThreadPoolExecutor()

        future_imu = None
        if self.has_imu:
            from toddlerbot.sensing.IMU import IMU

            future_imu = self.executor.submit(IMU)

        future_dynamixel = None
        if self.has_dynamixel:
            from toddlerbot.actuation.dynamixel.dynamixel_control import (
                DynamixelConfig,
                DynamixelController,
            )

            dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")

            dynamixel_ids = self.robot.get_joint_attrs("type", "dynamixel", "id")
            dynamixel_config = DynamixelConfig(
                port=dynamixel_ports[0],
                baudrate=self.robot.config["general"]["dynamixel_baudrate"],
                control_mode=self.robot.get_joint_attrs(
                    "type", "dynamixel", "control_mode"
                ),
                kP=self.robot.get_joint_attrs("type", "dynamixel", "kp_real"),
                kI=self.robot.get_joint_attrs("type", "dynamixel", "ki_real"),
                kD=self.robot.get_joint_attrs("type", "dynamixel", "kd_real"),
                kFF2=self.robot.get_joint_attrs("type", "dynamixel", "kff2_real"),
                kFF1=self.robot.get_joint_attrs("type", "dynamixel", "kff1_real"),
                init_pos=self.robot.get_joint_attrs("type", "dynamixel", "init_pos"),
            )
            future_dynamixel = self.executor.submit(
                DynamixelController, dynamixel_config, dynamixel_ids
            )

        future_sunny_sky = None
        if self.has_sunny_sky:
            from toddlerbot.actuation.sunny_sky.sunny_sky_control import (
                SunnySkyConfig,
                SunnySkyController,
            )

            sunny_sky_ports: List[str] = find_ports("Feather")

            sunny_sky_ids = self.robot.get_joint_attrs("type", "sunny_sky", "id")
            sunny_sky_config = SunnySkyConfig(
                port=sunny_sky_ports[0],
                kP=self.robot.get_joint_attrs("type", "sunny_sky", "kp_real"),
                kD=self.robot.get_joint_attrs("type", "sunny_sky", "kd_real"),
                i_ff=self.robot.get_joint_attrs("type", "sunny_sky", "i_ff_real"),
                gear_ratio=self.robot.get_joint_attrs(
                    "type", "sunny_sky", "gear_ratio"
                ),
                joint_limit=self.robot.get_joint_attrs(
                    "type", "sunny_sky", "joint_limit"
                ),
                init_pos=self.robot.get_joint_attrs("type", "sunny_sky", "init_pos"),
            )
            future_sunny_sky = self.executor.submit(
                SunnySkyController, sunny_sky_config, sunny_sky_ids
            )

        # Assign the results of futures to the attributes
        if future_sunny_sky is not None:
            self.sunny_sky_controller = future_sunny_sky.result()
        if future_dynamixel is not None:
            self.dynamixel_controller = future_dynamixel.result()
        if future_imu is not None:
            self.imu = future_imu.result()

        for _ in range(100):
            self.get_observation()

    # @profile()
    def process_motor_reading(self, results: Dict[str, Dict[int, JointState]]) -> Obs:
        motor_state_dict_unordered: Dict[str, JointState] = {}
        if self.has_dynamixel:
            dynamixel_state = results["dynamixel"]
            for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
                motor_id = self.robot.config["joints"][motor_name]["id"]
                motor_state_dict_unordered[motor_name] = dynamixel_state[motor_id]

        if self.has_sunny_sky:
            sunny_sky_state = results["sunny_sky"]
            for motor_name in self.robot.get_joint_attrs("type", "sunny_sky"):
                motor_id = self.robot.config["joints"][motor_name]["id"]
                motor_state_dict_unordered[motor_name] = sunny_sky_state[motor_id]

        time_curr = 0.0
        motor_pos = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_vel = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_tor = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        for i, motor_name in enumerate(self.robot.motor_ordering):
            if i == 0:
                time_curr = motor_state_dict_unordered[motor_name].time

            if motor_name in self.negated_motor_names:
                motor_pos[i] = -motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = -motor_state_dict_unordered[motor_name].vel
            else:
                motor_pos[i] = motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = motor_state_dict_unordered[motor_name].vel

            motor_tor[i] = abs(motor_state_dict_unordered[motor_name].tor)

        obs = Obs(
            time=time_curr,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
        )
        return obs

    def step(self):
        pass

    # @profile()
    def get_observation(self, retries: int = 0):
        results: Dict[str, Any] = {}
        futures: Dict[str, Any] = {}
        if self.has_dynamixel:
            # results["dynamixel"] = self.dynamixel_controller.get_motor_state(retries)
            futures["dynamixel"] = self.executor.submit(
                self.dynamixel_controller.get_motor_state, retries
            )

        if self.has_sunny_sky:
            # results["sunny_sky"] = self.sunny_sky_controller.get_motor_state()
            futures["sunny_sky"] = self.executor.submit(
                self.sunny_sky_controller.get_motor_state
            )

        if self.has_imu:
            # results["imu"] = self.imu.get_state()
            futures["imu"] = self.executor.submit(self.imu.get_state)

        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        obs = self.process_motor_reading(results)

        if self.has_imu:
            obs.ang_vel = np.array(results["imu"]["ang_vel"], dtype=np.float32)
            obs.euler = np.array(results["imu"]["euler"], dtype=np.float32)

        return obs

    # @profile()
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        # Directions are tuned to match the assembly of the robot.
        # joints_config = self.robot.config["joints"]

        motor_angles_updated: Dict[str, float] = {}
        for name, angle in motor_angles.items():
            if name in self.negated_motor_names:
                motor_angles_updated[name] = -angle
            else:
                motor_angles_updated[name] = angle

        if self.has_dynamixel:
            dynamixel_pos = [
                motor_angles_updated[k]
                for k in self.robot.get_joint_attrs("type", "dynamixel")
            ]
            self.executor.submit(
                self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
            )

        if self.has_sunny_sky:
            sunny_sky_pos = [
                motor_angles_updated[k]
                for k in self.robot.get_joint_attrs("type", "sunny_sky")
            ]
            self.executor.submit(
                self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
            )

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        if self.has_dynamixel:
            dynamixel_kps: List[float] = []
            for k in self.robot.get_joint_attrs("type", "dynamixel"):
                if k in motor_kps:
                    dynamixel_kps.append(motor_kps[k])
                else:
                    dynamixel_kps.append(self.robot.config["joints"][k]["kp_real"])

            self.executor.submit(self.dynamixel_controller.set_kp, dynamixel_kps)

    def close(self):
        if self.has_dynamixel:
            self.executor.submit(self.dynamixel_controller.close_motors)
        if self.has_sunny_sky:
            self.executor.submit(self.sunny_sky_controller.close_motors)
        if self.has_imu:
            self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)
