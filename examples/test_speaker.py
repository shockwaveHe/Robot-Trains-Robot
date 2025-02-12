import requests
import sounddevice as sd
import soundfile as sf
import soxr

from toddlerbot.sensing.speaker import Speaker

# This script is used to download an audio file, resample it to 44100 Hz, and play it back using the speaker on Jetson.

if __name__ == "__main__":
    # URL of the audio file (Replace with the actual URL)
    audio_url = (
        "https://www.cs.uic.edu/~troy/spring09/cs101/SoundFiles/BabyElephantWalk60.wav"
    )

    # Download the audio file
    audio_path = "/tmp/downloaded_audio.wav"

    print("Downloading audio file...")
    response = requests.get(audio_url)
    with open(audio_path, "wb") as file:
        file.write(response.content)

    print(f"Audio downloaded to {audio_path}")

    # Load the downloaded audio file
    data, samplerate = sf.read(audio_path)

    # Resample data to 44100 Hz if needed
    new_samplerate = 44100
    data_resampled = soxr.resample(data, samplerate, new_samplerate)

    # Play the resampled audio using the UACDemo device
    speaker = Speaker()
    sd.play(data_resampled, device=speaker.device)
    sd.wait()
