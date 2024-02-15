import time

import numpy as np
from leap_node import LeapNode
from pynput import keyboard

import toddleroid.actuation.dynamixel.leap_hand_utils as lhu


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


def main():
    listener = KeyboardListener()
    leap_hand = LeapNode()

    curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16), zeros=False)
    curr_pos = np.array(curr_pos)
    pos_stride = 0.2

    add_key_mappings = {
        "q": 0,
        "a": 1,
        "z": 2,
        "e": 3,
        "d": 4,
        "c": 5,
        "t": 6,
        "g": 7,
        "b": 8,
        "u": 9,
        "j": 10,
        "o": 11,
        "1": 12,
        "3": 13,
        "5": 14,
        "7": 15,
    }

    minus_key_mappings = {
        "w": 0,
        "s": 1,
        "x": 2,
        "r": 3,
        "f": 4,
        "v": 5,
        "y": 6,
        "h": 7,
        "n": 8,
        "i": 9,
        "k": 10,
        "p": 11,
        "2": 12,
        "4": 13,
        "6": 14,
        "8": 15,
    }

    frequency = 20
    while True:
        try:
            if listener.key in add_key_mappings:
                index = add_key_mappings[listener.key]
                curr_pos[index] += pos_stride
            elif listener.key in minus_key_mappings:
                index = minus_key_mappings[listener.key]
                curr_pos[index] -= pos_stride
            else:
                continue

            leap_hand.set_leap(curr_pos)
            print("Position: " + str(leap_hand.read_pos()))
        except KeyboardInterrupt:
            break

        listener.key = ""
        time.sleep(1 / frequency)


if __name__ == "__main__":
    main()
