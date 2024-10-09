import contextlib
import os
from typing import Dict

os.environ["SDL_VIDEODRIVER"] = "dummy"

with contextlib.redirect_stdout(None):
    import pygame

# Initialize Pygame
pygame.init()
# Initialize the joystick
pygame.joystick.init()

# TODO: Enum?
DECK_AXIS_MAPPING = {
    "left_joystick_vertical": 1,
    "left_joystick_horizontal": 0,
    "right_joystick_vertical": 3,
    "right_joystick_horizontal": 2,
    # 4, 5, 6, 7 are tracking pads
    "L2": 9,
    "R2": 8,
}
DECK_BUTTON_MAPPING = {
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
XBOX_AXIS_MAPPING = {
    "left_joystick_vertical": 1,
    "left_joystick_horizontal": 0,
    "right_joystick_vertical": 4,
    "right_joystick_horizontal": 3,
    "L2": 2,
    "R2": 5,
}
XBOX_BUTTON_MAPPING = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "L1": 4,
    "R1": 5,
    "view": 6,
    "menu": 7,
    # TODO: Add d-pad
}
STADIA_AXIS_MAPPING = {
    "left_joystick_vertical": 1,
    "left_joystick_horizontal": 0,
    "right_joystick_vertical": 3,
    "right_joystick_horizontal": 2,
    "L2": 4,
    "R2": 5,
}
STADIA_BUTTON_MAPPING = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "view": 4,
    "menu": 6,
    "L1": 9,
    "R1": 10,
    "d_pad_up": 11,
    "d_pad_down": 12,
    "d_pad_left": 13,
    "d_pad_right": 14,
}


class Joystick:
    def __init__(self, dead_zone: float = 0.1):
        self.dead_zone = dead_zone
        self.joystick_mapping = {
            "view": "stand",
            "menu": "log",
            "left_joystick_vertical": "walk_vertical",
            "left_joystick_horizontal": "walk_horizontal",
            "right_joystick_vertical": "squat",
            "right_joystick_horizontal": "turn",
            "d_pad_up": "lean_left",
            "d_pad_down": "lean_right",
            "d_pad_left": "twist_left",
            "d_pad_right": "twist_right",
            "Y": "look_up",
            "A": "look_down",
            "X": "look_left",
            "B": "look_right",
        }
        # List all input devices
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected.")
            return

        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            device_name = joystick.get_name().lower()

            if "xbox" in device_name or "x-box" in device_name:
                print("Detected: Microsoft Xbox Controller")
                self.axis_mapping = XBOX_AXIS_MAPPING
                self.button_mapping = XBOX_BUTTON_MAPPING
                self.joystick = joystick
                break
            elif "google" in device_name and "stadia" in device_name:
                print("Detected: Google Stadia Controller")
                self.axis_mapping = STADIA_AXIS_MAPPING
                self.button_mapping = STADIA_BUTTON_MAPPING
                self.joystick = joystick
                break
            elif "steam" in device_name and "deck" in device_name:
                print("Detected: Steam Deck Controller")
                self.axis_mapping = DECK_AXIS_MAPPING
                self.button_mapping = DECK_BUTTON_MAPPING
                self.joystick = joystick
                break
            else:
                print(f"Unsupported controller detected: {device_name}")

    def get_axis(self, axis_name: str) -> float:
        axis_id = self.axis_mapping[axis_name]
        return self.joystick.get_axis(axis_id)

    def get_button(self, button_name: str) -> float:
        button_id = self.button_mapping[button_name]
        return self.joystick.get_button(button_id)

    def get_controller_input(self) -> Dict[str, float]:
        # Process pygame events
        pygame.event.pump()

        control_inputs: Dict[str, float] = {}
        for key, task in self.joystick_mapping.items():
            if key in self.button_mapping:
                value = self.get_button(key)
                control_inputs[task] = 0.0 if abs(value) < self.dead_zone else value
            elif key in self.axis_mapping:
                value = self.get_axis(key)
                control_inputs[task] = 0.0 if abs(value) < self.dead_zone else value

        return control_inputs


if __name__ == "__main__":
    joystick = Joystick()
    while True:
        print(joystick.get_controller_input())
        pygame.time.wait(100)
