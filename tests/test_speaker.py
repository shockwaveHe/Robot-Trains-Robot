import sounddevice as sd
import soundfile as sf
from scipy.signal import resample

from toddlerbot.sensing.speaker import Speaker

# Load the audio file
data, samplerate = sf.read("/home/toddy/Downloads/test.wav")

# Resample data to 44100 Hz if needed
new_samplerate = 44100
data_resampled = resample(data, int(len(data) * new_samplerate / samplerate))

# Play the resampled audio using the UACDemo device
speaker = Speaker()
sd.play(data_resampled, device=speaker.device)
sd.wait()
