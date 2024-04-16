import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread

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
    find_feather_port,
)
from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import HumanoidRobot, JointState
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log

# my_logger = logging.getLogger("my_logger")
# my_logger.setLevel(logging.WARNING)


class RealWorld(BaseSim):
    def __init__(self, robot):
        super().__init__()
        self.name = "real_world"
        self.robot = robot

        self.negated_joint_names = [
            "left_hip_yaw",
            "right_hip_yaw",
            "right_hip_pitch",
            "left_ank_pitch",
        ]

        self.dynamixel_init_pos = np.radians([245, 180, 180, 287, 180, 180])
        # TODO: Replace the hard-coded gains
        self.dynamixel_config = DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kFF2=[0, 0, 0, 0, 0, 0],
            kFF1=[0, 0, 0, 0, 0, 0],
            kP=[400, 2400, 800, 400, 2400, 800],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[400, 800, 400, 400, 800, 400],
            current_limit=[700, 700, 700, 700, 700, 700],
            init_pos=self.dynamixel_init_pos,
            gear_ratio=np.array([19 / 21, 1, 1, 19 / 21, 1, 1]),
        )

        self.sunny_sky_config = SunnySkyConfig(port=find_feather_port(), kP=40, kD=50)
        # Temorarily hard-coded joint range for SunnySky
        joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}

        mighty_zap_init_pos = []
        for side, ids in robot.ankle2mighty_zap.items():
            mighty_zap_init_pos += robot.ankle_ik([0] * len(ids))

        self.mighty_zap_config = MightyZapConfig(
            port="/dev/tty.usbserial-0001", init_pos=mighty_zap_init_pos
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_dynamixel = executor.submit(
                DynamixelController,
                self.dynamixel_config,
                motor_ids=sorted(list(robot.dynamixel_joint2id.values())),
            )
            future_sunny_sky = executor.submit(
                SunnySkyController,
                self.sunny_sky_config,
                joint_range_dict=joint_range_dict,
            )
            future_mighty_zap = executor.submit(
                MightyZapController,
                self.mighty_zap_config,
                motor_ids=sorted(list(robot.mighty_zap_joint2id.values())),
            )

            # Assign the results of futures to the attributes
            self.dynamixel_controller = future_dynamixel.result()
            self.sunny_sky_controller = future_sunny_sky.result()
            self.mighty_zap_controller = future_mighty_zap.result()

        self.queues = {
            "dynamixel": queue.Queue(),
            "sunny_sky": queue.Queue(),
            "mighty_zap": queue.Queue(),
        }

        self.threads = {
            "dynamixel": Thread(target=self.command_worker, args=("dynamixel",)),
            "sunny_sky": Thread(target=self.command_worker, args=("sunny_sky",)),
            "mighty_zap": Thread(target=self.command_worker, args=("mighty_zap",)),
        }

        self.time_start = time.time()

        for name, thread in self.threads.items():
            log(
                "The command thread is initializing...",
                header="RealWorld",
                level="debug",
            )
            thread.start()

    def _negate_joint_angles(self, joint_angles):
        negated_joint_angles = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                negated_joint_angles[name] = -angle
            else:
                negated_joint_angles[name] = angle

        return negated_joint_angles

    # @profile
    def set_joint_angles(
        self, joint_angles, motor_list=["dynamixel", "sunny_sky", "mighty_zap"]
    ):
        # Directions are tuned to match the assembly of the robot.
        joint_angles = self._negate_joint_angles(joint_angles)

        if "dynamixel" in motor_list:
            dynamixel_pos = [
                joint_angles[k] for k in self.robot.dynamixel_joint2id.keys()
            ]
            log(
                f"{round_floats(dynamixel_pos, 4)}",
                header="Dynamixel",
                level="debug",
            )
        if "sunny_sky" in motor_list:
            sunny_sky_pos = [
                joint_angles[k] for k in self.robot.sunny_sky_joint2id.keys()
            ]
            log(
                f"{round_floats(sunny_sky_pos, 4)}",
                header="SunnySky",
                level="debug",
            )
        if "mighty_zap" in motor_list:
            for side, ids in self.robot.ankle2mighty_zap.items():
                ankle_pos = [
                    joint_angles[self.robot.mighty_zap_id2joint[id]] for id in ids
                ]
                mighty_zap_pos = self.robot.ankle_ik(ankle_pos)
                joint_angles[self.robot.mighty_zap_id2joint[ids[0]]] = mighty_zap_pos[0]
                joint_angles[self.robot.mighty_zap_id2joint[ids[1]]] = mighty_zap_pos[1]

            mighty_zap_pos = [
                joint_angles[k] for k in self.robot.mighty_zap_joint2id.keys()
            ]
            log(
                f"{round_floats(mighty_zap_pos, 1)}",
                header="MightyZap",
                level="debug",
            )

        # Execute set_pos calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            if "dynamixel" in motor_list:
                executor.submit(
                    self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
                )
            if "sunny_sky" in motor_list:
                executor.submit(
                    self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
                )
            if "mighty_zap" in motor_list:
                executor.submit(
                    self.mighty_zap_controller.set_pos, mighty_zap_pos, interp=False
                )

    def command_worker(self, actuator_type):
        while True:
            command = self.queues[actuator_type].get()
            if command is None:  # Use None as a signal to stop the thread
                break

            # Unpack the command
            command_type = command[0]
            # Here, based on command_type, call the appropriate method
            # For simplicity, assuming a 'set_pos' command type
            if command_type == "get_state":
                controller = getattr(self, f"{actuator_type}_controller")
                state = controller.get_motor_state()
                command[-1].set_result(state)
            else:
                log(
                    f"Unknown command type {command_type} for {actuator_type}",
                    level="error",
                )

            self.queues[actuator_type].task_done()

    # @profile
    def get_joint_state(self, motor_list=["dynamixel", "sunny_sky", "mighty_zap"]):
        if "dynamixel" in motor_list:
            future_dynamixel = Future()
            self.queues["dynamixel"].put(("get_state", None, future_dynamixel))
        if "sunny_sky" in motor_list:
            future_sunny_sky = Future()
            self.queues["sunny_sky"].put(("get_state", None, future_sunny_sky))
        if "mighty_zap" in motor_list:
            future_mighty_zap = Future()
            self.queues["mighty_zap"].put(("get_state", None, future_mighty_zap))

        dynamixel_state = sunny_sky_state = mighty_zap_state = None
        if "dynamixel" in motor_list:
            dynamixel_state = future_dynamixel.result()
        if "sunny_sky" in motor_list:
            sunny_sky_state = future_sunny_sky.result()
        if "mighty_zap" in motor_list:
            # Note: MightyZap positions are the lengthsmof linear actuators
            mighty_zap_state = future_mighty_zap.result()

        joint_state_dict = {}
        for name in self.robot.joints_info.keys():
            id = None
            if name in self.robot.dynamixel_joint2id and dynamixel_state is not None:
                id = self.robot.dynamixel_joint2id[name]
                joint_state_dict[name] = JointState(
                    time=dynamixel_state[id].time, pos=dynamixel_state[id].pos
                )
            elif name in self.robot.sunny_sky_joint2id and sunny_sky_state is not None:
                id = self.robot.sunny_sky_joint2id[name]
                joint_state_dict[name] = JointState(
                    time=sunny_sky_state[id].time, pos=sunny_sky_state[id].pos
                )
            elif (
                name in self.robot.mighty_zap_joint2id and mighty_zap_state is not None
            ):
                id = self.robot.mighty_zap_joint2id[name]
                joint_state_dict[name] = JointState(
                    time=mighty_zap_state[id].time, pos=mighty_zap_state[id].pos
                )

        return joint_state_dict

    def postprocess_ankle_pos(self, mighty_zap_pos_dict):
        ankle_pos_dict = {}
        for side, mighty_zap_pos_arr in mighty_zap_pos_dict.items():
            ids = self.robot.ankle2mighty_zap[side]
            last_ankle_pos = [0] * len(ids)
            ankle_pos_list = []
            for mighty_zap_pos in mighty_zap_pos_arr:
                ankle_pos = self.robot.ankle_fk(mighty_zap_pos, last_ankle_pos)
                ankle_pos_list.append(ankle_pos)
                last_ankle_pos = ankle_pos

            ankle_pos_arr = np.array(ankle_pos_list).T

            for i in range(len(ids)):
                ankle_pos_dict[self.robot.mighty_zap_id2joint[ids[i]]] = list(
                    ankle_pos_arr[i]
                )

        return ankle_pos_dict

    def get_link_pos(self, link_name: str):
        pass

    def get_torso_pose(self):
        return np.array([0, 0, self.robot.com[-1]]), np.eye(3)

    def get_zmp(self):
        pass

    def close(self):
        # Signal threads to stop
        for name in self.queues.keys():
            self.queues[name].put(None)
            self.threads[name].join()

        self.dynamixel_controller.close_motors()
        self.sunny_sky_controller.close_motors()
        self.mighty_zap_controller.close_motors()


if __name__ == "__main__":
    robot = HumanoidRobot("toddlerbot_legs")
    sim = RealWorld(robot)

    _, initial_joint_angles = robot.initialize_joint_angles()

    try:
        while True:
            sim.set_joint_angles(initial_joint_angles)
            joint_state_dict = sim.get_joint_state()
            log(f"Joint states: {joint_state_dict}", header="RealWorld", level="debug")
    finally:
        sim.close()
