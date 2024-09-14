import contextlib
from typing import List

import numpy as np

with contextlib.redirect_stdout(None):
    import pygame

from pygame.joystick import JoystickType


def initialize_joystick() -> JoystickType:
    # Initialize Pygame
    pygame.init()
    # Initialize the joystick
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    return joystick


def get_controller_input(
    joystick: JoystickType,
    command_ranges: List[List[float]],
    dead_zone: float = 0.05,
) -> list[float]:
    # Process pygame events
    pygame.event.pump()

    num_commands = len(command_ranges)
    axes = [0.0] * num_commands

    # Get joystick axes (assuming standard Xbox controller layout, adjust based on num_commands)
    if num_commands > 0:
        axes[0] = -joystick.get_axis(1)
    if num_commands > 1:
        axes[1] = -joystick.get_axis(0)
    if num_commands > 2:
        axes[2] = -joystick.get_axis(3)

    # Apply dead zones or thresholds for more precise control
    for i in range(num_commands):
        axes[i] = 0 if abs(axes[i]) < dead_zone else axes[i]

    # Scale the controller input to the range of the command
    controller_input: List[float] = []
    for i, value in enumerate(axes):
        if value < 0:
            scaled_value = np.interp(value, [-1, 0], [command_ranges[i][0], 0]).item()
        else:
            scaled_value = np.interp(value, [0, 1], [0, command_ranges[i][1]]).item()

        controller_input.append(scaled_value)

    return controller_input
