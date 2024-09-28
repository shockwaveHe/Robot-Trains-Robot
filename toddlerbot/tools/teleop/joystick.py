import contextlib
from typing import Dict, List

with contextlib.redirect_stdout(None):
    import pygame

from pygame.joystick import JoystickType

AXIS_MAPPING = {
    "left_joystick_vertical": 1,
    "left_joystick_horizontal": 0,
    "right_joystick_vertical": 3,
    "right_joystick_horizontal": 2,
    "left_trigger": 4,
    "right_trigger": 5,
}
BUTTON_MAPPING = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "left_bumper": 4,
    "right_bumper": 5,
    "back": 6,
    "start": 7,
    "L3": 8,
    "R3": 9,
    "d_pad_up": 10,
    "d_pad_down": 11,
    "d_pad_left": 12,
    "d_pad_right": 13,
}


class Joystick:
    def __init__(self, dead_zone: float = 0.05):
        self.dead_zone = dead_zone
        self.joystick = self.initialize_joystick()
        self.joystick_mapping = {
            "view": "stand",
            "d_pad_up": "neck_up",
            "d_pad_down": "neck_down",
            "d_pad_left": "neck_left",
            "d_pad_right": "neck_right",
            "left_joystick_vertical": "walk_vertical",
            "left_joystick_horizontal": "walk_horizontal",
            "right_joystick_vertical": "squat",
            "right_joystick_horizontal": "walk_turn",
            "A": "balance",
        }

    def get_axis(self, axis_name: str) -> float:
        if axis_name in AXIS_MAPPING:
            axis_id = AXIS_MAPPING[axis_name]
            return self.joystick.get_axis(axis_id)

    def get_button(self, button_name: str) -> bool:
        if button_name in BUTTON_MAPPING:
            button_id = BUTTON_MAPPING[button_name]
            return self.joystick.get_button(button_id)

    def initialize_joystick(self) -> JoystickType:
        # Initialize Pygame
        pygame.init()
        # Initialize the joystick
        pygame.joystick.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        return joystick

    def get_controller_input(self) -> Dict[str, float]:
        # Process pygame events
        pygame.event.pump()

        task_commands: Dict[str, float] = {}
        for key, task in self.joystick_mapping.items():
            if key in BUTTON_MAPPING:
                value = self.get_button(key)
                task_commands[task] = 0.0 if abs(value) < self.dead_zone else value
            elif key in AXIS_MAPPING:
                value = self.get_axis(key)
                task_commands[task] = 0.0 if abs(value) < self.dead_zone else value

        return task_commands


if __name__ == "__main__":
    joystick = Joystick()
    while True:
        print(joystick.get_controller_input())
        pygame.time.wait(100)
