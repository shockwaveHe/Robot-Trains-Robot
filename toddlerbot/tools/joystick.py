import contextlib
import os
import platform
from enum import Enum
from typing import Dict

import numpy as np

from toddlerbot.locomotion.mjx_config import get_env_config

os_type = platform.system()
if os_type != "Windows":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

with contextlib.redirect_stdout(None):
    import pygame

# Initialize Pygame
pygame.init()
# Initialize the joystick
pygame.joystick.init()


class DeckAxis(Enum):
    """Steam Deck Controller axis mapping."""

    LEFT_JOYSTICK_VERTICAL = 1
    LEFT_JOYSTICK_HORIZONTAL = 0
    RIGHT_JOYSTICK_VERTICAL = 3
    RIGHT_JOYSTICK_HORIZONTAL = 2
    L2 = 9
    R2 = 8


class DeckButton(Enum):
    """Steam Deck Controller button mapping."""

    A = 3
    B = 4
    X = 5
    Y = 6
    L1 = 7
    R1 = 8
    VIEW = 11
    MENU = 12
    DPAD_UP = 16
    DPAD_DOWN = 17
    DPAD_LEFT = 18
    DPAD_RIGHT = 19
    L4 = 20
    R4 = 21
    L5 = 22
    R5 = 23


class AsusAxis(Enum):
    """ROG Ally X Controller axis mapping."""

    LEFT_JOYSTICK_VERTICAL = 1
    LEFT_JOYSTICK_HORIZONTAL = 0
    RIGHT_JOYSTICK_VERTICAL = 3
    RIGHT_JOYSTICK_HORIZONTAL = 2
    L2 = 4
    R2 = 5


class AsusButton(Enum):
    """ROG Ally X Controller button mapping."""

    A = 0
    B = 1
    X = 2
    Y = 3
    L1 = 4
    R1 = 5
    VIEW = 6
    MENU = 7
    DPAD_UP = (0, 1)
    DPAD_DOWN = (0, -1)
    DPAD_LEFT = (-1, 0)
    DPAD_RIGHT = (1, 0)


class XboxAxis(Enum):
    """Xbox Controller axis mapping."""

    LEFT_JOYSTICK_VERTICAL = 1
    LEFT_JOYSTICK_HORIZONTAL = 0
    RIGHT_JOYSTICK_VERTICAL = 4
    RIGHT_JOYSTICK_HORIZONTAL = 3
    L2 = 2
    R2 = 5


class XboxButton(Enum):
    """Xbox Controller button mapping."""

    A = 0
    B = 1
    X = 2
    Y = 3
    L1 = 4
    R1 = 5
    VIEW = 6
    MENU = 7
    # Define the hat directions for D-pad
    DPAD_UP = (0, 1)
    DPAD_DOWN = (0, -1)
    DPAD_LEFT = (-1, 0)
    DPAD_RIGHT = (1, 0)


class StadiaAxis(Enum):
    """Google Stadia Controller axis mapping."""

    LEFT_JOYSTICK_VERTICAL = 1
    LEFT_JOYSTICK_HORIZONTAL = 0
    RIGHT_JOYSTICK_VERTICAL = 3
    RIGHT_JOYSTICK_HORIZONTAL = 2
    L2 = 4
    R2 = 5


class StadiaButton(Enum):
    """Google Stadia Controller button mapping."""

    A = 0
    B = 1
    X = 2
    Y = 3
    VIEW = 4
    MENU = 6
    L1 = 9
    R1 = 10
    DPAD_UP = 11
    DPAD_DOWN = 12
    DPAD_LEFT = 13
    DPAD_RIGHT = 14


class JoystickAction(Enum):
    """Joystick action mapping."""

    VIEW = "reset"
    MENU = "teleop"
    LEFT_JOYSTICK_VERTICAL = "walk_x"
    LEFT_JOYSTICK_HORIZONTAL = "walk_y"
    RIGHT_JOYSTICK_VERTICAL = "squat"
    RIGHT_JOYSTICK_HORIZONTAL = "walk_turn"
    DPAD_UP = "lean_left"
    DPAD_DOWN = "lean_right"
    DPAD_LEFT = "twist_left"
    DPAD_RIGHT = "twist_right"
    Y = "look_up"
    A = "look_down"
    X = "look_left"
    B = "look_right"
    L1 = "hug"
    R1 = "pick"
    L2 = "push_cart"
    R2 = "cuddle"
    L4 = "push_up"
    R4 = "dance_1"
    L5 = "dance_2"
    R5 = "dance_3"


class Joystick:
    """Joystick controller class for handling input from a connected joystick."""

    def __init__(self, dead_zone: float = 0.1):
        """Initializes the controller configuration by detecting and setting up a compatible joystick.

        Args:
            dead_zone (float): The threshold value for joystick input sensitivity, below which inputs are ignored. Defaults to 0.1.

        Raises:
            ValueError: If no joystick is detected or if an unsupported controller is detected.
        """
        self.dead_zone = dead_zone
        self.joystick = None

        walk_cfg = get_env_config("walk")
        walk_command_range = walk_cfg.commands.command_range
        self.walk_x_range = np.array(
            [walk_command_range[5][1], 0.0, walk_command_range[5][0]]
        )
        self.walk_y_range = np.array(
            [walk_command_range[6][1], 0.0, walk_command_range[6][0]]
        )
        self.walk_turn_range = np.array(
            [walk_command_range[7][1], 0.0, walk_command_range[7][0]]
        )
        # List all input devices
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise ValueError("No joystick detected.")
            return

        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            device_name = joystick.get_name().lower()

            if "xbox" in device_name or "x-box" in device_name:
                print("Detected: Microsoft Xbox Controller")
                self.axis_mapping: type[Enum] = XboxAxis
                self.button_mapping: type[Enum] = XboxButton
                self.joystick = joystick
                break
            elif "google" in device_name and "stadia" in device_name:
                print("Detected: Google Stadia Controller")
                self.axis_mapping = StadiaAxis
                self.button_mapping = StadiaButton
                self.joystick = joystick
                break
            elif "steam" in device_name and "deck" in device_name:
                print("Detected: Steam Deck Controller")
                self.axis_mapping = DeckAxis
                self.button_mapping = DeckButton
                self.joystick = joystick
                break
            elif "asus" in device_name:
                print("Detected: Asus Controller")
                self.axis_mapping = AsusAxis
                self.button_mapping = AsusButton
                self.joystick = joystick
                break
            else:
                raise ValueError(f"Unsupported controller detected: {device_name}")

    def get_controller_input(self) -> Dict[str, float]:
        """Retrieves and processes input from a connected joystick, mapping it to control actions.

        This function reads the current state of the joystick, including button presses and axis movements, and translates these inputs into a dictionary of control actions with corresponding float values. It handles both button and axis inputs, applying a dead zone threshold to filter out minor movements. Special handling is applied for D-pad directions and trigger buttons (L2, R2).

        Returns:
            Dict[str, float]: A dictionary where keys are control action names and values are the processed input values from the joystick.
        """
        pygame.event.pump()
        control_inputs: Dict[str, float] = {}

        if self.joystick is not None:
            for key in JoystickAction.__members__:
                task = JoystickAction[key].value
                if key in self.button_mapping.__members__:
                    # Handle button presses
                    button_id = self.button_mapping[key].value
                    if isinstance(button_id, tuple):
                        # Check if it's a D-pad direction
                        hat_x, hat_y = self.joystick.get_hat(0)
                        if (hat_x, hat_y) == button_id:
                            control_inputs[task] = 1.0
                    else:
                        value = float(self.joystick.get_button(button_id))
                        control_inputs[task] = (
                            0.0 if abs(value) < self.dead_zone else value
                        )
                elif key in self.axis_mapping.__members__:
                    # Handle axis motions
                    axis_id = self.axis_mapping[key].value
                    value = self.joystick.get_axis(axis_id)
                    if "L2" in key or "R2" in key:
                        value = 0 if value < 0.9 else 1

                    control_inputs[task] = 0.0 if abs(value) < self.dead_zone else value

        self.update_walk_command(control_inputs)

        return control_inputs

    def update_walk_command(self, control_inputs: Dict[str, float]):
        """Updates the walk command inputs by interpolating control values to specified ranges.

        Args:
            control_inputs (Dict[str, float]): A dictionary containing control tasks as keys
                ('walk_x', 'walk_y', 'walk_turn') and their corresponding input values. The input
                values are expected to be in the range [-1, 1].

        Modifies:
            control_inputs: The input values for each task are updated by interpolating them to
            the respective ranges defined by `self.walk_x_range`, `self.walk_y_range`, and
            `self.walk_turn_range`.
        """
        for task, input in control_inputs.items():
            if task == "walk_x":
                control_inputs[task] = np.interp(
                    control_inputs[task], [-1, 0, 1], self.walk_x_range
                ).item()
            elif task == "walk_y":
                control_inputs[task] = np.interp(
                    control_inputs[task], [-1, 0, 1], self.walk_y_range
                ).item()
            elif task == "walk_turn":
                control_inputs[task] = np.interp(
                    control_inputs[task], [-1, 0, 1], self.walk_turn_range
                ).item()


if __name__ == "__main__":
    joystick = Joystick()
    while True:
        print(joystick.get_controller_input())
        pygame.time.wait(100)
