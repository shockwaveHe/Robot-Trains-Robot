import sounddevice as sd


class Microphone:
    def __init__(self, mic_name="USB 2.0 Camera"):
        self.device = None
        for i, device in enumerate(sd.query_devices()):
            if mic_name in device["name"]:
                self.device = i
                print(f"Found microphone device at index: {i}")
                break
