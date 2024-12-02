from typing import Callable, Dict

from pynput import keyboard

keyboard_actions = {
    "save": "s",
    "next": "n",
}


class Keyboard:
    def __init__(self):
        self.key_inputs = {name: 0.0 for name in keyboard_actions.keys()}
        self.key_flags = {name: False for name in keyboard_actions.keys()}
        self.key_funcs = {}
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def register(self, name: str, func: Callable):
        if name in self.key_inputs and name not in self.key_funcs:
            self.key_funcs[name] = func

    def check(self, name: str, **kwargs):
        if self.key_inputs[name] == 1.0 and not self.key_flags[name]:
            # Append the current action to keyframes
            self.key_flags[name] = True
            self.key_funcs[name](**kwargs)

        elif self.key_inputs[name] == 0.0 and self.key_flags[name]:
            self.key_flags[name] = False

    def on_press(self, key):
        """Handles key press events."""
        try:
            self.key_inputs["speed_delta"] = 0.0
            self.key_inputs["force_delta"] = 0.0
            self.key_inputs["walk_x_delta"] = 0.0
            self.key_inputs["walk_y_delta"] = 0.0
            self.key_inputs["stop"] = False
            if key.char == "s":  # Check if the 's' key is pressed
                self.key_inputs["save"] = 1.0
                self.key_inputs["walk_x_delta"] = 0.01
            elif key.char == "n":
                self.key_inputs["next"] = 1.0
            elif key == keyboard.Key.right:
                self.key_inputs["speed_delta"] = 1.0
            elif key == keyboard.Key.left:
                self.key_inputs["speed_delta"] = -1.0
            elif key == keyboard.Key.up:
                self.key_inputs["force_delta"] = 1.0
            elif key == keyboard.Key.down:
                self.key_inputs["force_delta"] = -1.0
            elif key.char == "w":
                self.key_inputs["walk_x_delta"] = -0.01
            elif key.char == "a":
                self.key_inputs["walk_y_delta"] = 0.01
            elif key.char == "d":
                self.key_inputs["walk_y_delta"] = -0.01
            elif key == keyboard.Key.esc:
                # Signal the threads to stop
                self.key_inputs["stop"] = True
                return False  # Stop the keyboard listener
        except AttributeError:
            # Handle special keys (if necessary)
            pass

    def on_release(self, key):
        """Handles key release events."""
        try:
            if key.char == "s":  # Reset the 's' key state when released
                self.key_inputs["save"] = 0.0
            elif key.char == "n":
                self.key_inputs["next"] = 0.0
        except AttributeError:
            pass

    def close(self):
        """Stop the keyboard listener."""
        self.listener.stop()
        
    def get_keyboard_input(self) -> Dict[str, float]:
        """Return the current keyboard input state."""
        return self.key_inputs


if __name__ == "__main__":
    keyboard = Keyboard()
    try:
        while True:
            print(keyboard.get_keyboard_input())
    except KeyboardInterrupt:
        print("Exiting...")
