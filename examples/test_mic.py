import numpy as np
import sounddevice as sd

from toddlerbot.sensing.microphone import Microphone
from toddlerbot.sensing.speaker import Speaker

# Record and play back audio using the microphone and speaker on Jetson

if __name__ == "__main__":
    duration = 5  # seconds
    fs = 44100  # Sample rate

    mic = Microphone()
    print("Recording...")
    recorded_audio = sd.rec(
        int(duration * fs), samplerate=fs, channels=1, dtype=np.int16, device=mic.device
    )
    sd.wait()
    print("Playing back...")
    # Play the resampled audio using the UACDemo device
    speaker = Speaker()
    sd.play(recorded_audio, device=speaker.device)
    sd.wait()
