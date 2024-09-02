from typing import List, Optional

import numpy as np
import pygame
from pygame.joystick import JoystickType


def initialize_joystick():
    # Initialize Pygame
    try:
        pygame.init()
        # Initialize the joystick
        pygame.joystick.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    except Exception:
        joystick = None

    return joystick


def get_controller_input(
    joystick: JoystickType,
    command_ranges: Optional[List[List[float]]] = None,
    dead_zone: float = 0.05,
):
    # Process pygame events
    pygame.event.pump()

    # Get joystick axes (assuming standard Xbox controller)
    axis_0 = joystick.get_axis(0)  # Left stick horizontal (for linear velocity y)
    axis_1 = joystick.get_axis(1)  # Left stick vertical (for linear velocity x)
    axis_3 = joystick.get_axis(
        3
    )  # Right stick horizontal (for angular velocity z and heading direction)

    # Adjust axis values (e.g., invert axis if needed, apply scaling, etc.)
    lin_vel_x = -axis_1  # Inverting because pushing stick up gives negative values
    lin_vel_y = -axis_0  # Inverting because pushing stick left gives negative values
    ang_vel_yaw = axis_3

    # Apply dead zones or thresholds for more precise control
    lin_vel_x = 0 if abs(lin_vel_x) < dead_zone else lin_vel_x
    lin_vel_y = 0 if abs(lin_vel_y) < dead_zone else lin_vel_y
    ang_vel_yaw = 0 if abs(ang_vel_yaw) < dead_zone else ang_vel_yaw

    # Scale the controller input to the range of the command
    controller_input = [lin_vel_x, lin_vel_y, ang_vel_yaw]
    for i, value in enumerate(controller_input):
        if value < 0:
            controller_input[i] = np.interp(value, [-1, 0], [command_ranges[i][0], 0])  # type:ignore

        else:
            controller_input[i] = np.interp(value, [0, 1], [0, command_ranges[i][1]])  # type:ignore

    return controller_input


if __name__ == "__main__":
    try:
        joystick = initialize_joystick()
        assert joystick is not None, "No joystick found."

        while True:
            # Get the mapped controller input
            command = get_controller_input(joystick)

            # Print the command
            print(
                f"Command: [Linear X: {command[0]:.2f}, Linear Y: {command[1]:.2f}, Angular Z: {command[2]:.2f}]"
            )

            # Add a small delay to avoid spamming
            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        pygame.quit()
