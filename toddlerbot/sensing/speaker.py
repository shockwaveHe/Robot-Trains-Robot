import re
import subprocess

import sounddevice as sd


class Speaker:
    """Speaker class for handling audio output to a speaker device."""

    def __init__(self, speaker_name="UACDemo"):
        """Initializes the audio configuration by setting the volume of the specified speaker and identifying its device index.

        Args:
            speaker_name (str): The name of the speaker to configure. Defaults to "UACDemo".

        Raises:
            Exception: If an error occurs during the execution of system commands or device queries.
        """
        try:
            # Run `aplay -l` command to list audio cards
            result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, text=True)
            output = result.stdout

            # Use regex to find card numbers and names
            card_info = re.findall(r"card (\d+): (\w+)", output)
            speaker_card = None
            for card_num, card_name in card_info:
                if speaker_name in card_name:
                    speaker_card = str(card_num)
                    break

            if speaker_card is not None:
                subprocess.run(["amixer", "-c", speaker_card, "sset", "PCM", "100%"])

        except Exception as e:
            print(f"Error: {e}")

        self.device = None
        for i, device in enumerate(sd.query_devices()):
            if speaker_name in device["name"]:
                self.device = i
                print(f"Found speaker device at index: {i}")
                break
