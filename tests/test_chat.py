import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import io
import librosa

from openai import OpenAI
from toddlerbot.sensing.speaker import Speaker
from toddlerbot.sensing.microphone import Microphone

client = OpenAI()
speaker = Speaker()
microphone = Microphone()


# Function to capture audio and transcribe it
def capture_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=microphone.device) as source:
        print("Listening...")
        audio_data = recognizer.listen(source)
        print("Recognizing...")
        try:
            # Transcribe audio to text
            text = recognizer.recognize_google(audio_data)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError:
            print("Speech Recognition service is unavailable")
            return ""


# Function to get response from ChatGPT
def get_chatgpt_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cute humanoid robot called Toddy.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error fetching response from ChatGPT: {e}")
        return "Sorry, I couldn't process that."


# Function to convert text to speech and play it
def speak_text(text, volume=2.0, target_sr=44100, semitones=6):
    response = client.audio.speech.create(
        model="tts-1", voice="echo", response_format="wav", input=text
    )
    # Load the WAV data directly from the response into a BytesIO object
    wav_data = io.BytesIO(response.content)

    # Read and play the WAV audio data using sounddevice
    data, samplerate = sf.read(wav_data)

    data *= volume

    # data = librosa.effects.pitch_shift(data, sr=samplerate, n_steps=semitones)

    # Step 3: Resample to target sample rate if needed
    if samplerate != target_sr:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=target_sr)

    sd.play(data, device=speaker.device)
    sd.wait()


# Main function to run the chat agent
def chat_with_gpt():
    while True:
        user_input = capture_audio()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting chat.")
            break

        print(f"User: {user_input}")

        # Get response from ChatGPT
        response = get_chatgpt_response(user_input)
        print(f"ChatGPT: {response}")

        # Speak the response
        speak_text(response)


# Run the chat agent
chat_with_gpt()
