import contextlib
from typing import Dict

with contextlib.redirect_stdout(None):
    import pygame

from pygame.joystick import JoystickType

AXIS_MAPPING = {
    "left_joystick_vertical": 1,
    "left_joystick_horizontal": 0,
    "right_joystick_vertical": 3,
    "right_joystick_horizontal": 2,
    # 4, 5, 6, 7 are tracking pads
    "L2": 9,
    "R2": 8,
}
BUTTON_MAPPING = {
    "A": 3,
    "B": 4,
    "X": 5,
    "Y": 6,
    "L1": 7,
    "R1": 8,
    "view": 11,
    "menu": 12,
    "d_pad_up": 16,
    "d_pad_down": 17,
    "d_pad_left": 18,
    "d_pad_right": 19,
    "L4": 20,
    "R4": 21,
    "L5": 22,
    "R5": 23,
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

        control_inputs: Dict[str, float] = {}
        for key, task in self.joystick_mapping.items():
            if key in BUTTON_MAPPING:
                value = self.get_button(key)
                control_inputs[task] = 0.0 if abs(value) < self.dead_zone else value
            elif key in AXIS_MAPPING:
                value = self.get_axis(key)
                control_inputs[task] = 0.0 if abs(value) < self.dead_zone else value

        return control_inputs


if __name__ == "__main__":
    joystick = Joystick()
    while True:
        print(joystick.get_controller_input())
        pygame.time.wait(100)
