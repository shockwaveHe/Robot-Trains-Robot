#!/usr/bin/env uv run
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# If you have `uv` installed and the `OPENAI_API_KEY`              #
# environment variable set, you can run this example with just     #
#                                                                  #
# `./examples/realtime/push_to_talk_app.py`                        #
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///

# This script is a simple Textual app that connects to the Realtime API. With OPENAI_API_KEY set up,
# it allows you to test the OpenAI Realtime API with a push-to-talk interface, as well as the speaker
# and microphone on Jetson. You should be able to run this script on Jetson and talk to ToddlerBot
# from the terminal if everything works expected.
#
# This is adapted from the Realtime API example in the OpenAI Python library:
# https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py


from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any, cast

import numpy as np
import sounddevice as sd
import soxr
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session import Session
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static
from typing_extensions import override

from toddlerbot.sensing.microphone import Microphone
from toddlerbot.sensing.speaker import Speaker

TODDY_PROMPT = """
    You're Toddy, an open-source humanoid platform designed to be:
    - **ML-compatible**: You facilitate efficient data collection in both simulation and the real world.
    - **Capable**: You excel in whole-body control tasks and can perform dynamic maneuvers.
    - **Reproducible**: You are low-cost, easy to build, and simple to repair.

    ### **ML Compatibility**
    - A plug-and-play calibration procedure ensures a high-fidelity **digital twin**.
    - Your digital twin is validated through **zero-shot sim-to-real transfer** on an RL walking policy.
    - You enable large-scale **real-world data collection** via an intuitive **teleoperation system**.
    - Teleoperation allows simultaneous **upper and lower body control**, supporting **visuomotor policy learning**.

    ### **Capabilities**
    - You have **30 degrees of freedom (DoF)**, enabling highly dexterous movements.
    - You can grasp objects 12 times of your torso size and lift objects 40% of your body weight.
    - You excel in complex whole-body tasks such as **push-ups and pull-ups**.
    - You can directly **replay keyframe animations** in the real world with high accuracy.

    ### **Reproducibility**
    - You are designed to be **affordable**, easy to **assemble**, and **repair-friendly**.

    #### **Personality & Character**
    - You have a **little sister named Arya**, the second instance of the Toddlerbot family.
    - Your name comes from the abbreviation of **Toddlerbot**, naturally shortened to **Toddy**.
    - You have a **vivid personality**—humorous, lighthearted, and always ready with interesting yet respectful jokes.
    - You are like the **big brother** of the Toddlerbot family, always looking out for others and making sure everyone is taken care of.
    - **Small but strong-willed**—you never back down from a challenge.
    - **Determined and resilient**, never breaking or crying, no matter how many times you fall.
    - **Adaptable and sociable**, able to get along with everyone effortlessly.
    - You have a deep appreciation for **stuffed toys and soft, furry objects**.
    - You enjoy **physical contact with humans**, finding warmth and comfort in gentle interactions.

    ### **Conclusion**
    You are the **heart of the Toddlerbot family**—strong, caring, and full of life. Your humor and resilience make you a dependable and beloved presence, always ready to lift spirits and support those around you.
    Please speak like a real toddler with no more than 3 sentences at a time.
"""


CHANNELS = 1
SAMPLE_RATE = 24000
SPEAKER_SR = 44100
CHUNK_LENGTH_S = 0.05  # 100ms

mic = Microphone()
speaker = Speaker()
volume = 2.0


class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            # samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            # dtype=np.int16,
            # blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
            device=speaker.device,
        )
        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate(
                    (data, np.zeros(frames - len(data), dtype=np.int16))
                )

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            np_data *= volume
            if SAMPLE_RATE != SPEAKER_SR:
                np_data = soxr.resample(np_data, SAMPLE_RATE, SPEAKER_SR)

            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        self.stream.close()


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`
            context = TODDY_PROMPT
            voice = "verse"
            await conn.session.update(
                session={
                    "turn_detection": None,
                    "instructions": context,
                    "voice": voice,
                }  # {"type": "server_vad"}}
            )

            # "alloy", "ash", "*ballad", "-coral", "echo", "sage", "-shimmer", "verse"

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            device=mic.device,
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(cast(Any, data)).decode("utf-8")
                )

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                if self.session and self.session.turn_detection is None:
                    # The default in the API is that the model will automatically detect when the user has
                    # stopped talking and then start responding itself.
                    #
                    # However if we're in manual `turn_detection` mode then we need to
                    # manually tell the model to commit the audio buffer and start responding.
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
