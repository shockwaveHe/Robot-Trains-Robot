import time
from typing import List

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.misc_utils import log


def get_dynamixel_controller() -> DynamixelController:
    dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")

    dynamixel_ids: List[int] = [1, 2, 3, 4]
    control_mode: List[str] = ["extended_position", "extended_position"]
    kP: List[float] = [2400, 2400, 2400, 2400]
    kI: List[float] = [0, 0, 0, 0]
    kD: List[float] = [0, 0, 0, 0]
    kFF2: List[float] = [0, 0, 0, 0]
    kFF1: List[float] = [0, 0, 0, 0]
    init_pos: List[float] = [0, 0, 0, 0]

    dynamixel_config = DynamixelConfig(
        port=dynamixel_ports[0],
        baudrate=4000000,
        control_mode=control_mode,
        kP=kP,
        kI=kI,
        kD=kD,
        kFF2=kFF2,
        kFF1=kFF1,
        init_pos=init_pos,
    )
    dynamixel_controller = DynamixelController(dynamixel_config, dynamixel_ids)

    return dynamixel_controller


controller = get_dynamixel_controller()
step_idx = 0
step_time_list: List[float] = []
try:
    while True:
        step_start = time.time()

        motor_state = controller.get_motor_state()
        print(motor_state)

        step_idx += 1

        step_time = time.time() - step_start
        step_time_list.append(step_time)
        log(f"Latency: {step_time * 1000:.2f} ms", header="Test", level="debug")

except KeyboardInterrupt:
    pass

finally:
    time.sleep(1)

    controller.close_motors()

    log(
        f"Average Latency: {sum(step_time_list) / len(step_time_list) * 1000:.2f} ms",
        header="Test",
        level="info",
    )
