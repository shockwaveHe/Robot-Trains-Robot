import numpy as np

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.misc_utils import log, precise_sleep, profile


def main():
    controller = DynamixelController(
        DynamixelConfig(
            port=find_ports("USB <-> Serial Converter"),
            kFF2=[0] * 18,
            kFF1=[0] * 18,
            kP=[800] * 18,
            kI=[0] * 18,
            kD=[200] * 18,
            init_pos=np.radians(
                [
                    134,
                    99,
                    135,
                    180,
                    218,
                    45,
                    267,
                    171,
                    180,
                    341,
                    45,
                    180,
                    98,
                    90,
                    70,
                    175,
                    180,
                    251,
                ]
            ),
            gear_ratio=np.array([1] * 18),
            baudrate=4000000,
            default_vel=np.pi,
        ),
        motor_ids=list(range(0, 18)),
    )

    i = 0
    while i < 30:
        controller.set_pos(
            [
                np.pi / 6,
                np.pi / 6,
                np.pi / 12,
                -np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi * 9 / 6,
                np.pi / 12,
                -np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi / 12,
                np.pi * 9 / 6,
            ],
            vel=[
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi * 2,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi * 2,
            ],
        )

        i += 1

    i = 0
    while i < 30:
        controller.set_pos(
            [0.0] * 18,
            vel=[
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi * 2,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.pi * 2,
            ],
        )
        i += 1

    precise_sleep(0.1)
    controller.close_motors()

    log("Process completed successfully.", header="Dynamixel")


if __name__ == "__main__":
    main()
