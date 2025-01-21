import platform

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict, List

import numpy as np

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot

class RealWorldMock(BaseSim):
    """
    A debug version of the RealWorld class.
    All hardware calls are mocked; if there's a return value, it returns random data
    in the correct format. Otherwise, does nothing.
    """

    def __init__(self, robot: Robot):
        super().__init__("real_world_debug")
        self.robot = robot

        self.has_imu = self.robot.config["general"].get("has_imu", False)
        self.has_dynamixel = self.robot.config["general"].get("has_dynamixel", False)
        self.has_sunny_sky = self.robot.config["general"].get("has_sunny_sky", False)
 
        # Same motor names that need to be negated
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
        # Instead of setting up real hardware, just store a thread pool for parity
        self.executor = ThreadPoolExecutor()

        # Simulate discovering OS type
        self.os_type = platform.system()
        self.get_observation()
        # No real hardware initialization, so do nothing.
        pass

    def process_motor_reading(self, results: Dict[str, Dict[int, JointState]]) -> Obs:
        """
        Normally reads the dictionary of actual motor states from hardware.
        Here, we generate random states consistent with the shape/format.
        """
        # Create a dictionary of random JointStates
        motor_state_dict_unordered: Dict[str, JointState] = {}

        # If we "have" dynamixel, generate random data
        if self.has_dynamixel and "dynamixel" in results:
            for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
                mock_time = results["dynamixel"][motor_name].time
                mock_pos = results["dynamixel"][motor_name].pos
                mock_vel = results["dynamixel"][motor_name].vel
                mock_tor = results["dynamixel"][motor_name].tor
                motor_state_dict_unordered[motor_name] = JointState(
                    mock_time, mock_pos, mock_vel, mock_tor
                )

        # If we "have" sunny_sky, generate random data
        if self.has_sunny_sky and "sunny_sky" in results:
            for motor_name in self.robot.get_joint_attrs("type", "sunny_sky"):
                mock_time = results["sunny_sky"][motor_name].time
                mock_pos = results["sunny_sky"][motor_name].pos
                mock_vel = results["sunny_sky"][motor_name].vel
                mock_tor = results["sunny_sky"][motor_name].tor
                motor_state_dict_unordered[motor_name] = JointState(
                    mock_time, mock_pos, mock_vel, mock_tor
                )

        # For any motor not covered above, fill random data.
        for motor_name in self.robot.motor_ordering:
            if motor_name not in motor_state_dict_unordered:
                motor_state_dict_unordered[motor_name] = JointState(
                    time=np.random.random(), 
                    pos=np.random.uniform(-1.0, 1.0),
                    vel=np.random.uniform(-1.0, 1.0),
                    tor=np.random.uniform(-0.5, 0.5),
                )

        # Build arrays from the above data
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
            time=time.time() + 5.0,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
        )
        return obs

    def step(self):
        """Does nothing in the debug version."""
        pass

    def reset(self) -> Obs:
        """
        Returns a random observation that simulates a 'reset' in the real robot.
        """
        return self.get_observation()

    def get_observation(self, retries: int = 0) -> Obs:
        """
        In real hardware, we'd gather IMU and motor data. Here, we create random data.
        """
        results: Dict[str, Any] = {}

        # Create random data for dynamixel motors if we 'have' them
        if self.has_dynamixel:
            dynamixel_data: Dict[str, JointState] = {}
            for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
                dynamixel_data[motor_name] = JointState(
                    time=np.random.random(),
                    pos=np.random.uniform(-1.0, 1.0),
                    vel=np.random.uniform(-1.0, 1.0),
                    tor=np.random.uniform(-0.5, 0.5),
                )
            results["dynamixel"] = dynamixel_data

        # Create random data for sunny_sky motors if we 'have' them
        if self.has_sunny_sky:
            sunny_data: Dict[str, JointState] = {}
            for motor_name in self.robot.get_joint_attrs("type", "sunny_sky"):
                sunny_data[motor_name] = JointState(
                    time=np.random.random(),
                    pos=np.random.uniform(-2.0, 2.0),
                    vel=np.random.uniform(-2.0, 2.0),
                    tor=np.random.uniform(-1.0, 1.0),
                )
            results["sunny_sky"] = sunny_data

        # Create random IMU data if we 'have' an IMU
        if self.has_imu:
            # Just mock some typical 3D sensor readings
            imu_data = {
                "ang_vel": np.random.uniform(-1.0, 1.0, size=3).astype(np.float32),
                "euler": np.random.uniform(-3.14, 3.14, size=3).astype(np.float32),
            }
            results["imu"] = imu_data

        obs = self.process_motor_reading(results)

        # Insert IMU data into obs if present
        if self.has_imu:
            obs.ang_vel = results["imu"]["ang_vel"]
            obs.euler = results["imu"]["euler"]

        return obs

    def set_motor_target(self, motor_angles: Dict[str, float]):
        """
        In debug mode, do nothing except maybe log if desired. 
        We do not move real motors here.
        """
        pass

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """
        In debug mode, do nothing.
        """
        pass

    def is_done(self, obs):
        """
        Same logic as in BaseSim, or override if needed.
        """
        return super().is_done(obs)

    def close(self):
        """
        In debug mode, just shut down the executor or do nothing.
        """
        self.executor.shutdown(wait=True)
