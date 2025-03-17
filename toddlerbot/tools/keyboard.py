from typing import Callable, Dict

try:
    from pynput import keyboard
except ImportError:
    pass

keyboard_actions = {
    "save": "s",
    "next": "n",
    "walk_x_delta": None,
    "walk_y_delta": None,
    "stop": None,
    "speed_delta": None,
    "force_delta": None,
    "z_pos_delta": None,
    "pause": None,
    "resume": None,
}


class Keyboard:
    """A class for handling keyboard input events."""

    def __init__(self):
        """Initializes the KeyListener class, setting up dictionaries to track key inputs and flags, and starts a keyboard listener to handle key press and release events."""
        self.key_inputs = {name: 0.0 for name in keyboard_actions.keys()}
        self.key_flags = {name: False for name in keyboard_actions.keys()}
        self.key_funcs = {}
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def register(self, name: str, func: Callable):
        """Registers a function to a specified key if the key is already present in `key_inputs` but not in `key_funcs`.

        Args:
            name (str): The key associated with the function to be registered.
            func (Callable): The function to be registered under the specified key.
        """
        if name in self.key_inputs and name not in self.key_funcs:
            self.key_funcs[name] = func

    def check(self, name: str, **kwargs):
        """Checks and updates the state of a key input, triggering associated functions.

        This method evaluates the current state of a specified key input. If the key input is active and has not been flagged, it triggers the associated function and updates the flag. If the key input is inactive and has been flagged, it resets the flag.

        Args:
            name (str): The name of the key input to check.
            **kwargs: Additional keyword arguments to pass to the associated function.
        """
        if self.key_inputs[name] == 1.0 and not self.key_flags[name]:
            # Append the current action to keyframes
            self.key_flags[name] = True
            self.key_funcs[name](**kwargs)

        elif self.key_inputs[name] == 0.0 and self.key_flags[name]:
            self.key_flags[name] = False

    def reset(self):
        self.key_inputs["speed_delta"] = 0.0
        self.key_inputs["force_delta"] = 0.0
        self.key_inputs["walk_x_delta"] = 0.0
        self.key_inputs["walk_y_delta"] = 0.0
        self.key_inputs["z_pos_delta"] = 0.0
        self.key_inputs["stop"] = False
        self.key_inputs["pause"] = False
        self.key_inputs["resume"] = False

    def on_press(self, key):
        """Handles key press events."""
        # print(f"{key} pressed!")

        try:
            if key == keyboard.KeyCode.from_char("6"):
                print("change speed delta to 1.0")
                self.key_inputs["speed_delta"] = 1.0
            elif key == keyboard.KeyCode.from_char("4"):
                print("change speed delta to -1.0")
                self.key_inputs["speed_delta"] = -1.0
            elif key == keyboard.KeyCode.from_char("8"):
                self.key_inputs["force_delta"] = 1.0
            elif key == keyboard.KeyCode.from_char("2"):
                self.key_inputs["force_delta"] = -1.0
            elif key == keyboard.Key.esc:
                # Signal the threads to stop
                self.key_inputs["stop"] = True
                return False  # Stop the keyboard listener
            elif key.char == "n":
                self.key_inputs["next"] = 1.0
            elif key.char == "s":  # Check if the 's' key is pressed
                self.key_inputs["save"] = 1.0
                self.key_inputs["walk_y_delta"] = -0.01
            elif key.char == "a":
                self.key_inputs["walk_x_delta"] = 0.01
            elif key.char == "d":
                self.key_inputs["walk_x_delta"] = -0.01
            elif key.char == "w":
                self.key_inputs["walk_y_delta"] = 0.01
            elif key.char == "7":
                self.key_inputs["z_pos_delta"] = -0.01
            elif key.char == "9":
                self.key_inputs["z_pos_delta"] = 0.01
            elif key.char == "P":
                self.key_inputs["pause"] = True
            elif key.char == "R":
                self.key_inputs["resume"] = True
        except AttributeError:
            # Handle special keys (if necessary)
            pass

    def on_release(self, key):
        """Handle key release events to update key input states.

        This method is triggered when a key is released. It specifically checks for the 's' and 'n' keys and resets their corresponding states in the `key_inputs` dictionary to 0.0. If the released key does not have a character attribute, the exception is caught and ignored.

        Args:
            key: The key event object containing information about the released key.
        """
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
        # print(self.key_inputs)
        key_inputs = self.key_inputs
        return key_inputs


if __name__ == "__main__":
    keyboard = Keyboard()
    try:
        while True:
            print(keyboard.get_keyboard_input())
    except KeyboardInterrupt:
        print("Exiting...")
