import platform
import subprocess
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import BaseController, JointState
from toddlerbot.actuation.dynamixel.dynamixel_client import DynamixelClient

# from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log  # profile

CONTROL_MODE_DICT: Dict[str, int] = {
    "current": 0,
    "velocity": 1,
    "position": 3,
    "extended_position": 4,
    "current_based_position": 5,
    "pwm": 16,
}


@dataclass
class DynamixelConfig:
    port: str
    baudrate: int
    control_mode: List[str]
    kP: List[float]
    kI: List[float]
    kD: List[float]
    kFF2: List[float]
    kFF1: List[float]
    init_pos: List[float]
    default_vel: float = np.pi
    interp_method: str = "cubic"
    return_delay_time: int = 1


class DynamixelController(BaseController):
    def __init__(self, config: DynamixelConfig, motor_ids: List[int]):
        super().__init__()

        self.config = config
        self.motor_ids: List[int] = motor_ids
        self.lock = Lock()

        self.connect_to_client()
        self.initialize_motors()

        if len(self.config.init_pos) == 0:
            self.init_pos = np.zeros(len(motor_ids), dtype=np.float32)
        else:
            self.init_pos = np.array(config.init_pos, dtype=np.float32)
            self.update_init_pos()

    def connect_to_client(self, latency_value: int = 1):
        os_type = platform.system()
        try:
            if os_type == "Linux":
                # Construct the command to set the latency timer on Linux
                command = f"echo {latency_value} | sudo tee /sys/bus/usb-serial/devices/{self.config.port.split('/')[-1]}/latency_timer"
            elif os_type == "Darwin":
                # Construct the command to set the latency timer on macOS
                command = f"./toddlerbot/actuation/dynamixel/latency_timer_setter_macOS/set_latency_timer -l {latency_value}"
            else:
                raise Exception(f"Unsupported OS: {os_type}")
            # Run the command
            result = subprocess.run(
                command, shell=True, text=True, check=True, stdout=subprocess.PIPE
            )
            log(f"Latency Timer set: {result.stdout.strip()}", header="Dynamixel")
        except subprocess.CalledProcessError as e:
            log(f"Failed to set latency timer: {e}", header="Dynamixel", level="error")

        time.sleep(0.1)

        try:
            self.client = DynamixelClient(
                self.motor_ids, self.config.port, self.config.baudrate
            )
            self.client.connect()
            log(f"Connected to the port: {self.config.port}", header="Dynamixel")

        except Exception:
            raise ConnectionError("Could not connect to the Dynamixel port.")

    def initialize_motors(self):
        log("Initializing motors...", header="Dynamixel")
        # Set the return delay time to 1*2=2us
        self.client.sync_write(
            self.motor_ids, [self.config.return_delay_time] * len(self.motor_ids), 9, 1
        )
        self.client.sync_write(
            self.motor_ids,
            [CONTROL_MODE_DICT[m] for m in self.config.control_mode],
            11,
            1,
        )
        self.client.sync_write(self.motor_ids, self.config.kD, 80, 2)
        self.client.sync_write(self.motor_ids, self.config.kI, 82, 2)
        self.client.sync_write(self.motor_ids, self.config.kP, 84, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF2, 88, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF1, 90, 2)
        # self.client.sync_write(self.motor_ids, self.config.current_limit, 102, 2)

        time.sleep(0.2)

        self.client.set_torque_enabled(self.motor_ids, True)

        time.sleep(0.2)

    def update_init_pos(self):
        _, pos_arr = self.client.read_pos(retries=-1)
        delta_pos = pos_arr - self.init_pos
        delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi
        self.init_pos = pos_arr - delta_pos

    def close_motors(self):
        open_clients: List[DynamixelClient] = list(DynamixelClient.OPEN_CLIENTS)  # type: ignore
        for open_client in open_clients:
            if open_client.port_handler.is_using:
                log("Forcing client to close.", header="Dynamixel")
            open_client.port_handler.is_using = False
            open_client.disconnect()

    def reboot_motors(self):
        self.client.reboot(self.motor_ids)

    def set_kp(self, kP: List[float]):
        self.client.sync_write(self.motor_ids, kP, 84, 2)

    # @profile()
    def set_pos(
        self,
        pos: List[float],
        interp: bool = True,
        vel: List[float] = [],
        delta_t: float = -1,
    ):
        def set_pos_helper(pos_arr: npt.NDArray[np.float32]):
            pos_arr_drive = self.init_pos + pos_arr
            with self.lock:
                self.client.write_desired_pos(self.motor_ids, pos_arr_drive)  # type: ignore

        pos_arr: npt.NDArray[np.float32] = np.array(pos)

        # if interp:
        #     pos_arr_start: npt.NDArray[np.float32] = np.array(
        #         [state.pos for state in self.get_motor_state().values()]
        #     )
        #     if len(vel) == 0 and delta_t < 0:
        #         delta_t = max(np.abs(pos_arr - pos_arr_start) / self.config.default_vel)
        #     elif delta_t < 0:
        #         delta_t = max(np.abs(pos_arr - pos_arr_start) / np.array(vel))

        #     interpolate_pos(
        #         set_pos_helper,
        #         pos_arr_start,
        #         pos_arr,
        #         delta_t,
        #         self.config.interp_method,
        #     )
        # else:

        set_pos_helper(pos_arr)

    # @profile()
    def get_motor_state(self, retries: int = 0) -> Dict[int, JointState]:
        # log(f"Start... {time.time()}", header="Dynamixel", level="warning")

        state_dict: Dict[int, JointState] = {}
        with self.lock:
            # time, pos_arr = self.client.read_pos(retries=retries)
            time, pos_arr, vel_arr = self.client.read_pos_vel(retries=retries)
            # time, pos_arr, vel_arr, cur_arr = self.client.read_pos_vel_cur(
            #     retries=retries
            # )

        # log(f"Pos: {np.round(pos_arr, 4)}", header="Dynamixel", level="debug")  # type: ignore
        # log(f"Vel: {np.round(vel_arr, 4)}", header="Dynamixel", level="debug")  # type: ignore
        # log(f"Cur: {np.round(cur_arr, 4)}", header="Dynamixel", level="debug")  # type: ignore

        # self.waist_act_1_max_current = max(
        #     self.waist_act_1_max_current, abs(cur_arr[2])
        # )
        # self.waist_act_2_max_current = max(
        #     self.waist_act_2_max_current, abs(cur_arr[3])
        # )
        # print(
        #     f"Max current: {self.waist_act_1_max_current:.2f}, {self.waist_act_2_max_current:.2f}"
        # )

        pos_arr -= self.init_pos

        for i, motor_id in enumerate(self.motor_ids):
            state_dict[motor_id] = JointState(time=time, pos=pos_arr[i], vel=vel_arr[i])

        # log(f"End... {time.time()}", header="Dynamixel", level="warning")

        return state_dict
