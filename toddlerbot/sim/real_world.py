from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, profile, snake2camel


class RealWorld(BaseSim):
    def __init__(self, robot: Robot, debug: bool = False):
        super().__init__()
        self.name = "real_world"
        self.robot = robot
        self.debug = debug

        self.has_imu = self.robot.config["general"]["has_imu"]
        self.has_dynamixel = self.robot.config["general"]["has_dynamixel"]
        self.has_sunny_sky = self.robot.config["general"]["has_sunny_sky"]

        # TODO: Update the negated joint names
        self.negated_joint_names: List[str] = []

        self._initialize()

    def _initialize(self):
        self.last_state: Dict[str, JointState] = {}

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
                gear_ratio=self.robot.get_joint_attrs(
                    "type", "dynamixel", "gear_ratio"
                ),
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

    def _negate_joint_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        joint_angles_negated: Dict[str, float] = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                joint_angles_negated[name] = -angle
            else:
                joint_angles_negated[name] = angle

        return joint_angles_negated

    # @profile()
    def set_joint_angles(self, joint_angles: Dict[str, float]):
        # Directions are tuned to match the assembly of the robot.
        joint_angles_negated = self._negate_joint_angles(joint_angles)

        left_ankle_pos, right_ankle_pos = self.robot.get_ankle_pos(joint_angles_negated)
        if len(left_ankle_pos) > 0:
            left_ank_motor_pos = self.robot.ankle_ik(left_ankle_pos)
            joint_angles_negated.update(left_ank_motor_pos)

        if len(right_ankle_pos) > 0:
            right_ank_motor_pos = self.robot.ankle_ik(right_ankle_pos)
            joint_angles_negated.update(right_ank_motor_pos)

        if self.has_dynamixel:
            dynamixel_pos = [
                joint_angles_negated[k]
                for k in self.robot.get_joint_attrs("type", "dynamixel")
            ]
            if self.debug:
                log(
                    f"{round_floats(dynamixel_pos, 4)}",
                    header="Dynamixel",
                    level="debug",
                )
            self.executor.submit(
                self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
            )

        if self.has_sunny_sky:
            sunny_sky_pos = [
                joint_angles_negated[k]
                for k in self.robot.get_joint_attrs("type", "sunny_sky")
            ]
            if self.debug:
                log(
                    f"{round_floats(sunny_sky_pos, 4)}",
                    header="SunnySky",
                    level="debug",
                )

            self.executor.submit(
                self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
            )

    def _finite_diff_vel(
        self, pos: float, last_pos: float, time: float, last_time: float
    ) -> float:
        return (pos - last_pos) / (time - last_time)

    # @profile()
    def _process_joint_state(
        self, results: Dict[str, Dict[int, JointState]]
    ) -> Dict[str, JointState]:
        joint_state_dict: Dict[str, JointState] = {}

        if self.has_dynamixel:
            dynamixel_state = results["dynamixel"]
            for joint_name in self.robot.get_joint_attrs("type", "dynamixel"):
                motor_id = self.robot.config["joints"][joint_name]["id"]
                has_closed_loop = self.robot.config["joints"][joint_name][
                    "has_closed_loop"
                ]
                if has_closed_loop:
                    if joint_name in self.last_state:
                        # TODO: Implement ankle and waist IK
                        dynamixel_state[motor_id].vel = self._finite_diff_vel(
                            dynamixel_state[motor_id].pos,
                            self.last_state[joint_name].pos,
                            dynamixel_state[motor_id].time,
                            self.last_state[joint_name].time,
                        )

                    self.last_state[joint_name] = dynamixel_state[motor_id]

                joint_state_dict[joint_name] = dynamixel_state[motor_id]

        if self.has_sunny_sky:
            sunny_sky_state = results["sunny_sky"]
            for joint_name in self.robot.get_joint_attrs("type", "sunny_sky"):
                motor_id = self.robot.config["joints"][joint_name]["id"]
                if joint_name in self.last_state:
                    sunny_sky_state[motor_id].vel = self._finite_diff_vel(
                        sunny_sky_state[motor_id].pos,
                        self.last_state[joint_name].pos,
                        sunny_sky_state[motor_id].time,
                        self.last_state[joint_name].time,
                    )

                self.last_state[joint_name] = sunny_sky_state[motor_id]
                joint_state_dict[joint_name] = sunny_sky_state[motor_id]

        # TODO: Calibrate the direction of the motors
        for joint_name in joint_state_dict.keys():
            if joint_name in self.negated_joint_names:
                joint_state_dict[joint_name].pos *= -1
                joint_state_dict[joint_name].vel *= -1

        return joint_state_dict

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
        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        joint_state_dict = self._process_joint_state(results)

        return joint_state_dict

    # @profile()
    def get_observation(
        self,
    ) -> Tuple[Dict[str, JointState], Dict[str, npt.NDArray[np.float32]]]:
        futures: Dict[str, Any] = {}
        if self.has_dynamixel:
            futures["dynamixel"] = self.executor.submit(
                self.dynamixel_controller.get_motor_state
            )

        if self.has_sunny_sky:
            futures["sunny_sky"] = self.executor.submit(
                self.sunny_sky_controller.get_motor_state
            )

        if self.has_imu:
            futures["imu"] = self.executor.submit(self.imu.get_state)

        results: Dict[str, Any] = {}
        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        joint_state_dict = self._process_joint_state(results)

        if self.has_imu:
            root_state: Dict[str, npt.NDArray[np.float32]] = results["imu"]
        else:
            root_state = {}

        return joint_state_dict, root_state

    def close(self):
        if self.has_dynamixel:
            self.executor.submit(self.dynamixel_controller.close_motors)
        if self.has_sunny_sky:
            self.executor.submit(self.sunny_sky_controller.close_motors)
        if self.has_imu:
            self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)
