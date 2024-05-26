import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.actuation.mighty_zap.mighty_zap_control import (
    MightyZapConfig,
    MightyZapController,
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
    def __init__(self, robot, debug=False):
        super().__init__()
        self.name = "real_world"
        self.robot = robot
        self.debug = debug

        self.negated_joint_names = [
            "left_hip_yaw",
            "right_hip_yaw",
            "left_ank_pitch",
        ]

        self._initialize()

    def _initialize(self):
        dynamixel_port = find_ports("USB <-> Serial Converter")
        sunny_sky_port = find_ports("Feather")
        mighty_zap_port = find_ports("USB Quad_Serial")

        n_ports = 2 + len(mighty_zap_port) + 1
        self.executor = ThreadPoolExecutor(max_workers=n_ports)

        future_imu = self.executor.submit(IMU)

        self.mighty_zap_ids = sorted(list(self.robot.mighty_zap_joint2id.values()))
        self.mighty_zap_init_pos = [0] * len(self.mighty_zap_ids)
        self.last_mighty_zap_state = None
        self.last_ankle_state = None

        mighty_zap_config = MightyZapConfig(
            port=mighty_zap_port, init_pos=self.mighty_zap_init_pos
        )
        future_mighty_zap = self.executor.submit(
            MightyZapController,
            mighty_zap_config,
            motor_ids=self.mighty_zap_ids,
        )

        dynamixel_ids = sorted(list(self.robot.dynamixel_joint2id.values()))
        # The init pos needs to be calibrated after the robot is assembled
        dynamixel_init_pos = np.radians([241.17, 180, 180, 118.12, 180, 180])
        # TODO: Replace the hard-coded gains
        dynamixel_config = DynamixelConfig(
            port=dynamixel_port,
            kFF2=[0, 0, 0, 0, 0, 0],
            kFF1=[0, 0, 0, 0, 0, 0],
            kP=[800, 3200, 1600, 800, 3200, 1600],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[800, 1600, 1600, 800, 1600, 1600],
            # current_limit=[700, 700, 700, 700, 700, 700],
            init_pos=dynamixel_init_pos,
            gear_ratio=np.array([19 / 21, 1, 1, 19 / 21, 1, 1]),
        )
        future_dynamixel = self.executor.submit(
            DynamixelController, dynamixel_config, motor_ids=dynamixel_ids
        )

        sunny_sky_config = SunnySkyConfig(port=sunny_sky_port)
        # Temorarily hard-coded joint range for SunnySky
        joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}
        self.sunny_sky_ids = sorted(list(self.robot.sunny_sky_joint2id.values()))
        self.last_sunny_sky_state = None
        future_sunny_sky = self.executor.submit(
            SunnySkyController,
            sunny_sky_config,
            joint_range_dict=joint_range_dict,
        )

        # Assign the results of futures to the attributes

        self.sunny_sky_controller = future_sunny_sky.result()
        self.dynamixel_controller = future_dynamixel.result()
        self.mighty_zap_controller = future_mighty_zap.result()
        self.imu = future_imu.result()

    def _negate_joint_angles(self, joint_angles):
        negated_joint_angles = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                negated_joint_angles[name] = -angle
            else:
                negated_joint_angles[name] = angle

        return negated_joint_angles

    # @profile()
    def set_joint_angles(self, joint_angles):
        # Directions are tuned to match the assembly of the robot.
        joint_angles = self._negate_joint_angles(joint_angles)

        dynamixel_pos = [joint_angles[k] for k in self.robot.dynamixel_joint2id.keys()]
        sunny_sky_pos = [joint_angles[k] for k in self.robot.sunny_sky_joint2id.keys()]
        mighty_zap_pos = []
        for side, motor_ids in self.robot.ankle2mighty_zap.items():
            ankle_pos = [
                joint_angles[self.robot.mighty_zap_id2joint[id]] for id in motor_ids
            ]
            mighty_zap_pos += self.robot.ankle_ik(ankle_pos)

        if self.debug:
            log(f"{round_floats(dynamixel_pos, 4)}", header="Dynamixel", level="debug")
            log(f"{round_floats(sunny_sky_pos, 4)}", header="SunnySky", level="debug")
            log(f"{round_floats(mighty_zap_pos, 1)}", header="MightyZap", level="debug")

        # Execute set_pos calls in parallel
        self.executor.submit(
            self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
        )
        self.executor.submit(
            self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
        )
        for mighty_zap_id, pos in zip(self.mighty_zap_ids, mighty_zap_pos):
            self.executor.submit(
                self.mighty_zap_controller.set_pos_single, pos, mighty_zap_id
            )

    @profile()
    def _process_joint_state(self, results):
        # Note: MightyZap positions are the lengthsmof linear actuators
        mighty_zap_state = {}
        for motor_name, result in results.items():
            if motor_name.startswith("mighty_zap"):
                motor_id = int(motor_name.split("_")[2])
                mighty_zap_state[motor_id] = result

        self.last_mighty_zap_state = copy.deepcopy(mighty_zap_state)
        ankle_state = copy.deepcopy(mighty_zap_state)

        sunny_sky_state = results["sunny_sky"]
        if self.last_sunny_sky_state is not None:
            for motor_id in self.sunny_sky_ids:
                pos_delta = (
                    sunny_sky_state[motor_id].pos
                    - self.last_sunny_sky_state[motor_id].pos
                )
                time_delta = (
                    sunny_sky_state[motor_id].time
                    - self.last_sunny_sky_state[motor_id].time
                )

                sunny_sky_state[motor_id].vel = pos_delta / time_delta

                if sunny_sky_state[motor_id].vel > 5:
                    log(
                        f"{motor_id}: {sunny_sky_state[motor_id].vel}="
                        + f"{pos_delta}/{time_delta}",
                        header=snake2camel(self.name),
                        level="warning",
                    )

        self.last_sunny_sky_state = copy.deepcopy(sunny_sky_state)

        dynamixel_state = results["dynamixel"]

        for side, motor_ids in self.robot.ankle2mighty_zap.items():
            mighty_zap_pos = []
            for motor_id in motor_ids:
                if (
                    mighty_zap_state[motor_id].pos > 0
                    and mighty_zap_state[motor_id].pos < 4096
                ):
                    mighty_zap_pos.append(mighty_zap_state[motor_id].pos)
                else:
                    mighty_zap_pos.append(
                        self.mighty_zap_init_pos[self.mighty_zap_ids.index(motor_id)]
                        if self.last_mighty_zap_state is None
                        else np.clip(self.last_mighty_zap_state[motor_id].pos, 1, 4095)
                    )
                    log(
                        f"The MightyZap position is out of range. Use {mighty_zap_pos[-1]}.",
                        header=snake2camel(self.name),
                        level="warning",
                    )

            ankle_pos = self.robot.ankle_fk(mighty_zap_pos)
            if np.any(np.isnan(ankle_pos)):
                ankle_state = self.last_ankle_state
            else:
                for i, motor_id in enumerate(motor_ids):
                    ankle_state[motor_id].pos = ankle_pos[i]
                    if self.last_ankle_state is not None:
                        pos_delta = ankle_pos[i] - self.last_ankle_state[motor_id].pos
                        time_delta = (
                            ankle_state[motor_id].time
                            - self.last_ankle_state[motor_id].time
                        )
                        ankle_state[motor_id].vel = pos_delta / time_delta

                        if ankle_state[motor_id].vel > 5:
                            log(
                                f"{motor_id}: {ankle_state[motor_id].vel}="
                                + f"{pos_delta}/{time_delta}",
                                header=snake2camel(self.name),
                                level="warning",
                            )

        self.last_ankle_state = copy.deepcopy(ankle_state)

        joint_state_dict = {}
        for name in self.robot.joints_info.keys():
            id = None
            if name in self.robot.dynamixel_joint2id and dynamixel_state is not None:
                id = self.robot.dynamixel_joint2id[name]
                joint_state_dict[name] = dynamixel_state[id]
            elif name in self.robot.sunny_sky_joint2id and sunny_sky_state is not None:
                id = self.robot.sunny_sky_joint2id[name]
                joint_state_dict[name] = sunny_sky_state[id]
            elif name in self.robot.mighty_zap_joint2id and ankle_state is not None:
                id = self.robot.mighty_zap_joint2id[name]
                joint_state_dict[name] = ankle_state[id]

        for joint_name in joint_state_dict.keys():
            if joint_name in self.negated_joint_names:
                joint_state_dict[joint_name].pos *= -1
                joint_state_dict[joint_name].vel *= -1

        return joint_state_dict

    # @profile()
    def get_joint_state(self):
        futures = {}
        for mighty_zap_id in self.mighty_zap_ids:
            futures[f"mighty_zap_{mighty_zap_id}"] = self.executor.submit(
                self.mighty_zap_controller.get_motor_state_single, mighty_zap_id
            )
        futures["dynamixel"] = self.executor.submit(
            self.dynamixel_controller.get_motor_state
        )
        futures["sunny_sky"] = self.executor.submit(
            self.sunny_sky_controller.get_motor_state
        )

        results = {}
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

    @profile()
    def get_observation(self):
        futures = {}
        for mighty_zap_id in self.mighty_zap_ids:
            futures[f"mighty_zap_{mighty_zap_id}"] = self.executor.submit(
                self.mighty_zap_controller.get_motor_state_single, mighty_zap_id
            )
        futures["dynamixel"] = self.executor.submit(
            self.dynamixel_controller.get_motor_state
        )
        futures["sunny_sky"] = self.executor.submit(
            self.sunny_sky_controller.get_motor_state
        )
        futures["imu"] = self.executor.submit(self.imu.get_state)

        results = {}
        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        joint_state_dict = self._process_joint_state(results)

        root_state = results["imu"]

        # root_state["quaternion"] = np.array([1, 0, 0, 0])
        # root_state["angular_velocity"] = np.array([0, 0, 0])

        return joint_state_dict, root_state

    def get_link_pos(self, link_name: str):
        pass

    def get_torso_pose(self):
        return np.array([0, 0, self.robot.com[-1]]), np.eye(3)

    def get_zmp(self):
        pass

    def close(self):
        self.executor.submit(self.dynamixel_controller.close_motors)
        self.executor.submit(self.sunny_sky_controller.close_motors)
        self.executor.submit(self.mighty_zap_controller.close_motors)
        self.executor.submit(self.imu.close)

        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    robot = HumanoidRobot("toddlerbot_legs")
    sim = RealWorld(robot)

    _, initial_joint_angles = robot.initialize_joint_angles()

    try:
        while True:
            sim.set_joint_angles(initial_joint_angles)
            joint_state_dict = sim.get_joint_state()
            log(
                f"Joint states: {joint_state_dict}",
                header=snake2camel(sim.name),
                level="debug",
            )
    finally:
        sim.close()
