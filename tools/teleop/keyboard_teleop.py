import numpy as np
from pynput import keyboard

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.utils.misc_utils import precise_sleep


class KeyboardListener:
    def __init__(self):
        self.key = ""
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        try:
            if key.char.isalnum():
                self.key = key.char
        except AttributeError:
            pass


def get_key_mappings(n_motors):
    add_keys = "1234567890"[:n_motors]
    minus_keys = "qwertyuiop"[:n_motors]

    add_key_mappings = {key: i for i, key in enumerate(add_keys)}
    minus_key_mappings = {key: i for i, key in enumerate(minus_keys)}

    print("Key Mappings for Increasing Position:", add_key_mappings)
    print("Key Mappings for Decreasing Position:", minus_key_mappings)

    return add_key_mappings, minus_key_mappings


def pretty_print_positions(target_pos_rad, actual_pos_rad):
    # Convert radians to degrees for user-friendly output
    target_pos_deg = np.degrees(target_pos_rad)
    actual_pos_deg = np.degrees(actual_pos_rad)

    # Creating formatted strings for target and actual positions
    target_pos_str = ", ".join(f"{pos:.2f}°" for pos in target_pos_deg)
    actual_pos_str = ", ".join(f"{pos:.2f}°" for pos in actual_pos_deg)

    print("Position:\n" f"  Target: {target_pos_str}\n" f"  Actual: {actual_pos_str}")


def main():
    n_motors = 3
    init_pos = np.radians([135, 180, 180])
    dynamixel_node = DynamixelController(DynamixelConfig(), n_motors, init_pos)

    listener = KeyboardListener()

    add_key_mappings, minus_key_mappings = get_key_mappings(n_motors)
    pos_stride = np.radians(5)
    frequency = 20  # Frequency of loop execution

    try:
        while True:
            key_pressed = listener.key
            if key_pressed:
                index = None
                if listener.key in add_key_mappings:
                    index = add_key_mappings[listener.key]
                elif listener.key in minus_key_mappings:
                    index = minus_key_mappings[listener.key]

                if index is not None:
                    dynamixel_node.curr_pos[index] += (
                        pos_stride if listener.key in add_key_mappings else -pos_stride
                    )
                    dynamixel_node.set_pos(dynamixel_node.curr_pos)
                    pretty_print_positions(
                        dynamixel_node.curr_pos, dynamixel_node.read_pos()
                    )

            listener.key = ""
            precise_sleep(1 / frequency)

    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")


if __name__ == "__main__":
    main()
