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

    default_pos = np.zeros(18)
    lower_limit = [
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
    ]
    upper_limit = [
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        0,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
    ]

    w = 5
    for joint_idx in range(w, w + 1):
        default_pos_copy = default_pos.copy()
        for limit in [lower_limit, upper_limit, default_pos]:
            default_pos_copy[joint_idx] = limit[joint_idx]
            i = 0
            while i < 30:
                controller.set_pos(default_pos_copy)
                i += 1

    # i = 0
    # while i < 30:
    #     controller.set_pos(
    #         [0.0] * 18,
    #         vel=[
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi * 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi / 2,
    #             np.pi * 2,
    #         ],
    #     )
    #     i += 1

    precise_sleep(0.1)

    controller.close_motors()

    log("Process completed successfully.", header="Dynamixel")


if __name__ == "__main__":
    main()
