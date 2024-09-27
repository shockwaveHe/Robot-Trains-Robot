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
    command_list: List[List[float]],
    dead_zone: float = 0.05,
) -> List[float]:
    # Process pygame events
    pygame.event.pump()

    num_commands = len(command_list[0])
    axes = [0.0] * num_commands
    # Get joystick axes (assuming standard Xbox controller layout, adjust based on num_commands)
    if num_commands > 0:
        axes[0] = -joystick.get_axis(1)
    if num_commands > 1:
        axes[1] = -joystick.get_axis(0)
    if num_commands > 2:
        axes[2] = -joystick.get_axis(3)

    for i in range(num_commands):
        if abs(axes[i]) < dead_zone:
            axes[i] = 0.0  # Zero command

    for command in command_list:
        if np.all(np.sign(axes) == np.sign(command)):
            return command

    # If no exact match is found, return a zero command or a fallback
    return [0.0] * num_commands
