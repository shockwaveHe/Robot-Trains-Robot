import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

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

        self._initialize_motors()
        self._initialize_sensors()

    def _initialize_motors(self):
        dynamixel_port = find_ports("USB <-> Serial Converter")
        sunny_sky_port = find_ports("Feather")
        mighty_zap_port = find_ports("USB Quad_Serial")

        n_ports = 1 + 1 + len(mighty_zap_port)

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
            current_limit=[700, 700, 700, 700, 700, 700],
            init_pos=dynamixel_init_pos,
            gear_ratio=np.array([19 / 21, 1, 1, 19 / 21, 1, 1]),
        )

        sunny_sky_config = SunnySkyConfig(port=sunny_sky_port)
        # Temorarily hard-coded joint range for SunnySky
        joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}

        self.mighty_zap_ids = sorted(list(self.robot.mighty_zap_joint2id.values()))
        self.mighty_zap_init_pos = [0] * len(self.mighty_zap_ids)
        self.last_mighty_zap_state = None

        mighty_zap_config = MightyZapConfig(
            port=mighty_zap_port, init_pos=self.mighty_zap_init_pos
        )

        self.executor = ThreadPoolExecutor(max_workers=n_ports)

        future_dynamixel = self.executor.submit(
            DynamixelController, dynamixel_config, motor_ids=dynamixel_ids
        )
        future_sunny_sky = self.executor.submit(
            SunnySkyController,
            sunny_sky_config,
            joint_range_dict=joint_range_dict,
        )
        future_mighty_zap = self.executor.submit(
            MightyZapController,
            mighty_zap_config,
            motor_ids=self.mighty_zap_ids,
        )

        # Assign the results of futures to the attributes
        self.dynamixel_controller = future_dynamixel.result()
        self.sunny_sky_controller = future_sunny_sky.result()
        self.mighty_zap_controller = future_mighty_zap.result()

    def _negate_joint_angles(self, joint_angles):
        negated_joint_angles = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                negated_joint_angles[name] = -angle
            else:
                negated_joint_angles[name] = angle

        return negated_joint_angles

    def _initialize_sensors(self):
        self.imu = IMU(default_pose=euler2quat(-np.pi / 2, 0, 0))

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

    # @profile()
    def get_joint_state(self):
        dynamixel_state = sunny_sky_state = mighty_zap_state = None

        # mighty_zap_state = {}
        # for mighty_zap_id in self.mighty_zap_ids:
        #     mighty_zap_state[mighty_zap_id] = (
        #         self.mighty_zap_controller.get_motor_state_single(mighty_zap_id)
        #     )

        # sunny_sky_state = self.sunny_sky_controller.get_motor_state()
        # dynamixel_state = self.dynamixel_controller.get_motor_state()

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

        # Note: MightyZap positions are the lengthsmof linear actuators
        mighty_zap_state = {}
        for motor_name, result in results.items():
            if motor_name.startswith("mighty_zap"):
                motor_id = int(motor_name.split("_")[2])
                mighty_zap_state[motor_id] = result

        sunny_sky_state = results["sunny_sky"]
        dynamixel_state = results["dynamixel"]

        for side, motor_ids in self.robot.ankle2mighty_zap.items():
            mighty_zap_pos = []
            for motor_id in motor_ids:
                if mighty_zap_state[motor_id].pos > 0:
                    mighty_zap_pos.append(mighty_zap_state[motor_id].pos)
                else:
                    log(
                        "The MightyZap position is negative",
                        header=snake2camel(self.name),
                        level="warning",
                    )
                    mighty_zap_pos.append(
                        self.mighty_zap_init_pos[self.mighty_zap_ids.index(motor_id)]
                        if self.last_mighty_zap_state is None
                        else self.last_mighty_zap_state[motor_id].pos
                    )

            ankle_pos = self.robot.ankle_fk(mighty_zap_pos)
            for i, motor_id in enumerate(motor_ids):
                mighty_zap_state[motor_id].pos = ankle_pos[i]
                if self.last_mighty_zap_state is not None:
                    mighty_zap_state[motor_id].vel = (
                        ankle_pos[i] - self.last_mighty_zap_state[motor_id].pos
                    ) / (
                        mighty_zap_state[motor_id].time
                        - self.last_mighty_zap_state[motor_id].time
                    )
                    # log(
                    #     f"{motor_id}: {mighty_zap_state[motor_id].vel}="
                    #     + f"({ankle_pos[i]}-{self.last_mighty_zap_state[motor_id].pos})/"
                    #     + f"({mighty_zap_state[motor_id].time}-{self.last_mighty_zap_state[motor_id].time})",
                    #     header=snake2camel(self.name),
                    #     level="debug",
                    # )

        self.last_mighty_zap_state = copy.deepcopy(mighty_zap_state)

        joint_state_dict = {}
        for name in self.robot.joints_info.keys():
            id = None
            if name in self.robot.dynamixel_joint2id and dynamixel_state is not None:
                id = self.robot.dynamixel_joint2id[name]
                joint_state_dict[name] = dynamixel_state[id]
            elif name in self.robot.sunny_sky_joint2id and sunny_sky_state is not None:
                id = self.robot.sunny_sky_joint2id[name]
                joint_state_dict[name] = sunny_sky_state[id]
            elif (
                name in self.robot.mighty_zap_joint2id and mighty_zap_state is not None
            ):
                id = self.robot.mighty_zap_joint2id[name]
                joint_state_dict[name] = mighty_zap_state[id]

        for joint_name in joint_state_dict.keys():
            if joint_name in self.negated_joint_names:
                joint_state_dict[joint_name].pos *= -1
                joint_state_dict[joint_name].vel *= -1

        # log(
        #     f"Joint states: {joint_state_dict}",
        #     header=snake2camel(self.name),
        #     level="debug",
        # )

        return joint_state_dict

    def get_observation(self, joint_ordering):
        joint_state_dict = self.get_joint_state()
        q = np.array([joint_state_dict[j].pos for j in joint_ordering])
        dq = np.array([joint_state_dict[j].vel for j in joint_ordering])
        quat = self.imu.get_quaternion()
        omega = self.imu.get_angular_velocity()
        return (q, dq, quat, omega)

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
