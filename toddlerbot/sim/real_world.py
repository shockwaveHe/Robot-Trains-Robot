from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.actuation.sunny_sky.sunny_sky_control import (
    SunnySkyConfig,
    SunnySkyController,
)
from toddlerbot.sensing.IMU import IMU
from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, profile, snake2camel


class RealWorld(BaseSim):
    def __init__(self, robot: HumanoidRobot, debug: bool = False):
        super().__init__()
        self.name = "real_world"
        self.robot = robot
        self.debug = debug

        self.negated_joint_names: List[str] = [
            "left_hip_yaw",
            "right_hip_yaw",
            "left_ank_pitch",
        ]

        self._initialize()

    def _initialize(self):
        dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")
        sunny_sky_ports: List[str] = find_ports("Feather")

        n_ports = len(dynamixel_ports) + len(sunny_sky_ports) + 1  # 1 is for IMU
        self.executor = ThreadPoolExecutor(max_workers=n_ports)

        future_imu = self.executor.submit(IMU)

        dynamixel_ids = self.robot.get_attrs("dynamixel", "id")
        dynamixel_config = DynamixelConfig(
            port=dynamixel_ports[0],
            control_mode=self.robot.get_attrs("dynamixel", "control_mode"),
            kP=self.robot.get_attrs("dynamixel", "kp_real"),
            kI=self.robot.get_attrs("dynamixel", "ki_real"),
            kD=self.robot.get_attrs("dynamixel", "kd_real"),
            kFF2=self.robot.get_attrs("dynamixel", "kff2_real"),
            kFF1=self.robot.get_attrs("dynamixel", "kff1_real"),
            gear_ratio=self.robot.get_attrs("dynamixel", "gear_ratio"),
            init_pos=self.robot.get_attrs("dynamixel", "init_pos"),
        )
        future_dynamixel = self.executor.submit(
            DynamixelController, dynamixel_config, dynamixel_ids
        )

        sunny_sky_ids = self.robot.get_attrs("sunny_sky", "id")
        sunny_sky_config = SunnySkyConfig(
            port=sunny_sky_ports[0],
            kP=self.robot.get_attrs("sunny_sky", "kp_real"),
            kD=self.robot.get_attrs("sunny_sky", "kd_real"),
            i_ff=self.robot.get_attrs("sunny_sky", "i_ff_real"),
            gear_ratio=self.robot.get_attrs("sunny_sky", "gear_ratio"),
            joint_limit=self.robot.get_attrs("sunny_sky", "joint_limit"),
            init_pos=self.robot.get_attrs("sunny_sky", "init_pos"),
        )
        future_sunny_sky = self.executor.submit(
            SunnySkyController, sunny_sky_config, sunny_sky_ids
        )

        self.last_state: Dict[str, JointState] = {}

        # Assign the results of futures to the attributes
        self.sunny_sky_controller = future_sunny_sky.result()
        self.dynamixel_controller = future_dynamixel.result()
        self.imu = future_imu.result()

    def _negate_joint_angles(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        negated_joint_angles: Dict[str, float] = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                negated_joint_angles[name] = -angle
            else:
                negated_joint_angles[name] = angle

        return negated_joint_angles

    # @profile()
    def set_joint_angles(self, joint_angles: Dict[str, float]):
        # Directions are tuned to match the assembly of the robot.
        joint_angles = self._negate_joint_angles(joint_angles)

        dynamixel_pos = [joint_angles[k] for k in self.robot.get_names("dynamixel")]
        sunny_sky_pos = [joint_angles[k] for k in self.robot.get_names("sunny_sky")]

        if self.debug:
            log(f"{round_floats(dynamixel_pos, 4)}", header="Dynamixel", level="debug")
            log(f"{round_floats(sunny_sky_pos, 4)}", header="SunnySky", level="debug")

        # Execute set_pos calls in parallel
        self.executor.submit(
            self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
        )
        self.executor.submit(
            self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
        )

    def _finite_diff_vel(
        self, pos: float, last_pos: float, time: float, last_time: float
    ) -> float:
        pos_delta = pos - last_pos
        time_delta = time - last_time
        return pos_delta / time_delta

    # @profile()
    def _process_joint_state(
        self, results: Dict[str, Dict[int, JointState]]
    ) -> Dict[str, JointState]:
        joint_state_dict: Dict[str, JointState] = {}
        sunny_sky_state = results["sunny_sky"]
        for joint_name in self.robot.get_names("sunny_sky"):
            motor_id = self.robot.config[joint_name]["id"]
            if joint_name in self.last_state:
                sunny_sky_state[motor_id].vel = self._finite_diff_vel(
                    sunny_sky_state[motor_id].pos,
                    self.last_state[joint_name].pos,
                    sunny_sky_state[motor_id].time,
                    self.last_state[joint_name].time,
                )

            self.last_state[joint_name] = sunny_sky_state[motor_id]
            joint_state_dict[joint_name] = sunny_sky_state[motor_id]

        dynamixel_state = results["dynamixel"]
        for joint_name in self.robot.get_names("dynamixel"):
            motor_id = self.robot.config[joint_name]["id"]
            if self.robot.config[joint_name]["has_closed_loop"]:
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

        # TODO: Calibrate the direction of the motors
        for joint_name in joint_state_dict.keys():
            if joint_name in self.negated_joint_names:
                joint_state_dict[joint_name].pos *= -1
                joint_state_dict[joint_name].vel *= -1

        return joint_state_dict

    def get_torso_pose(self):
        return np.array([0, 0, self.robot.com[-1]]), np.eye(3)

    # @profile()
    def get_joint_state(self) -> Dict[str, JointState]:
        futures: Dict[str, Any] = {}
        futures["dynamixel"] = self.executor.submit(
            self.dynamixel_controller.get_motor_state
        )
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
        futures["dynamixel"] = self.executor.submit(
            self.dynamixel_controller.get_motor_state
        )
        futures["sunny_sky"] = self.executor.submit(
            self.sunny_sky_controller.get_motor_state
        )
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

        root_state: Dict[str, npt.NDArray[np.float32]] = results["imu"]

        return joint_state_dict, root_state

    def close(self):
        self.executor.submit(self.dynamixel_controller.close_motors)
        self.executor.submit(self.sunny_sky_controller.close_motors)
        self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)
