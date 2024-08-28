from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, state_to_obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.math_utils import exponential_moving_average

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
            "left_hip_yaw_drive",
            "right_hip_yaw_drive",
            "left_knee_act",
            "right_knee_act",
            "left_sho_roll",
            "right_sho_roll",
            "left_sho_yaw_drive",
            "right_sho_yaw_drive",
            "left_elbow_roll",
            "right_elbow_roll",
            "left_elbow_yaw_drive",
            "right_elbow_yaw_drive",
        ]

        self.initialize()

    def initialize(self):
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

        self.last_joint_state_dict: Dict[str, JointState] = {}

        for _ in range(100):
            self.get_observation()

        self.last_joint_state_dict: Dict[str, JointState] = {}

    def negate_motor_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        joint_angles_negated: Dict[str, float] = {}
        for name, angle in joint_angles.items():
            if name in self.negated_motor_names:
                joint_angles_negated[name] = -angle
            else:
                joint_angles_negated[name] = angle

        return joint_angles_negated

    # @profile()
    def process_motor_reading(
        self, results: Dict[str, Dict[int, JointState]]
    ) -> Tuple[Dict[str, JointState], Dict[str, JointState]]:
        motor_state_dict: Dict[str, JointState] = {}

        if self.has_dynamixel:
            dynamixel_state = results["dynamixel"]
            for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
                motor_id = self.robot.config["joints"][motor_name]["id"]
                motor_state_dict[motor_name] = dynamixel_state[motor_id]

        if self.has_sunny_sky:
            sunny_sky_state = results["sunny_sky"]
            for motor_name in self.robot.get_joint_attrs("type", "sunny_sky"):
                motor_id = self.robot.config["joints"][motor_name]["id"]
                motor_state_dict[motor_name] = sunny_sky_state[motor_id]

        for motor_name in motor_state_dict.keys():
            if motor_name in self.negated_motor_names:
                motor_state_dict[motor_name].pos *= -1
                motor_state_dict[motor_name].vel *= -1

        motor_state_dict = {
            motor_name: motor_state_dict[motor_name]
            for motor_name in self.robot.motor_ordering
        }

        joint_state_dict: Dict[str, JointState] = {}
        joints_config = self.robot.config["joints"]
        waist_act_pos: List[float] = []
        left_ank_act_pos: List[float] = []
        right_ank_act_pos: List[float] = []
        for motor_name, motor_state in motor_state_dict.items():
            transmission = joints_config[motor_name]["transmission"]
            if transmission == "gears":
                joint_name = motor_name.replace("_drive", "_driven")
                joint_state_dict[joint_name] = JointState(
                    time=motor_state.time,
                    pos=motor_state.pos * joints_config[motor_name]["gear_ratio"],
                    vel=-motor_state.vel / joints_config[motor_name]["gear_ratio"],
                )
                if joint_name in self.negated_motor_names:
                    joint_state_dict[joint_name].vel *= -1

            elif transmission == "waist":
                # Placeholder to ensure the correct order
                joint_state_dict["waist_roll"] = JointState(
                    time=motor_state.time, pos=0.0, vel=0.0
                )
                joint_state_dict["waist_yaw"] = JointState(
                    time=motor_state.time, pos=0.0, vel=0.0
                )
                waist_act_pos.append(motor_state.pos)
            elif transmission == "knee":
                joint_name: str = motor_name.replace("_act", "_pitch")
                joint_state_dict[joint_name] = JointState(
                    time=motor_state.time, pos=motor_state.pos, vel=motor_state.vel
                )
            elif transmission == "ankle":
                if "left" in motor_name:
                    joint_state_dict["left_ank_roll"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    joint_state_dict["left_ank_pitch"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    left_ank_act_pos.append(motor_state.pos)
                elif "right" in motor_name:
                    joint_state_dict["right_ank_roll"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    joint_state_dict["right_ank_pitch"] = JointState(
                        time=motor_state.time, pos=0.0, vel=0.0
                    )
                    right_ank_act_pos.append(motor_state.pos)
            elif transmission == "none":
                joint_state_dict[motor_name] = JointState(
                    time=motor_state.time, pos=motor_state.pos, vel=motor_state.vel
                )

        joint_state_dict["waist_roll"].pos, joint_state_dict["waist_yaw"].pos = (
            self.robot.waist_fk(waist_act_pos)
        )
        (
            joint_state_dict["left_ank_roll"].pos,
            joint_state_dict["left_ank_pitch"].pos,
        ) = self.robot.ankle_fk(left_ank_act_pos, "left")
        (
            joint_state_dict["right_ank_roll"].pos,
            joint_state_dict["right_ank_pitch"].pos,
        ) = self.robot.ankle_fk(right_ank_act_pos, "right")

        for joint_name in [
            "waist_roll",
            "waist_yaw",
            "left_ank_roll",
            "left_ank_pitch",
            "right_ank_roll",
            "right_ank_pitch",
        ]:
            if self.last_joint_state_dict and joint_name in self.last_joint_state_dict:
                time_delta = (
                    joint_state_dict[joint_name].time
                    - self.last_joint_state_dict[joint_name].time
                )
                if time_delta > 0:
                    joint_state_dict[joint_name].vel = (
                        joint_state_dict[joint_name].pos
                        - self.last_joint_state_dict[joint_name].pos
                    ) / time_delta
                else:
                    raise ValueError("Time delta must be greater than 0.")

        for joint_name in joint_state_dict.keys():
            joint_state_dict[joint_name].vel = float(
                exponential_moving_average(
                    self.robot.config["general"]["fd_smooth_alpha"],
                    joint_state_dict[joint_name].vel,
                    self.last_joint_state_dict[joint_name].vel,
                )
            )

        self.last_joint_state_dict = joint_state_dict

        return motor_state_dict, joint_state_dict

    def get_torso_pose(self):
        return np.array([0, 0, 0.4]), np.eye(3)

    # @profile()
    def get_joint_state(self, retries: int = 0) -> Dict[str, JointState]:
        futures: Dict[str, Any] = {}
        if self.has_dynamixel:
            futures["dynamixel"] = self.executor.submit(
                self.dynamixel_controller.get_motor_state, retries
            )

        if self.has_sunny_sky:
            futures["sunny_sky"] = self.executor.submit(
                self.sunny_sky_controller.get_motor_state
            )

        results: Dict[str, Dict[int, JointState]] = {}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        _, joint_state_dict = self.process_motor_reading(results)

        return joint_state_dict

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

        motor_state_dict, joint_state_dict = self.process_motor_reading(results)

        obs = state_to_obs(motor_state_dict, joint_state_dict)

        if self.has_imu:
            for k, v in results["imu"].items():
                setattr(obs, k, v.copy())

        return obs

    # @profile()
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        # Directions are tuned to match the assembly of the robot.
        motor_angles_negated = self.negate_motor_angles(motor_angles)
        if self.has_dynamixel:
            dynamixel_pos = [
                motor_angles_negated[k]
                for k in self.robot.get_joint_attrs("type", "dynamixel")
            ]
            self.executor.submit(
                self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
            )

        if self.has_sunny_sky:
            sunny_sky_pos = [
                motor_angles_negated[k]
                for k in self.robot.get_joint_attrs("type", "sunny_sky")
            ]
            self.executor.submit(
                self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
            )

    def close(self):
        if self.has_dynamixel:
            self.executor.submit(self.dynamixel_controller.close_motors)
        if self.has_sunny_sky:
            self.executor.submit(self.sunny_sky_controller.close_motors)
        if self.has_imu:
            self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)
