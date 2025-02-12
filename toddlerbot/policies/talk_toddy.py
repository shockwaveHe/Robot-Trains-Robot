import asyncio
import base64
import threading
import time
from typing import Any, Dict, cast

import numpy as np
import numpy.typing as npt
import sounddevice as sd
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session import Session

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.replay import ReplayPolicy
from toddlerbot.sensing.microphone import Microphone
from toddlerbot.sensing.speaker import Speaker
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.audio_player_async import AudioPlayerAsync
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import interpolate_action

CHANNELS = 1
SAMPLE_RATE = 24000

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

    **Only speak the text inside the double quotes (””).**
    Do NOT read any instructions, character names, descriptions, the text inside the brackets ([]), or anything other than the lines. 
    Wait for Arya to respond before moving to the next sentence.

    Script:
    Toddy (confident, enthusiastic): "Of course! How many should I do?"
    Toddy (determined, ready): “Alright, here we go!”
    Toddy (grinning, catching her breath): Phew! Hope that looked right!”
"""


def start_async_task(task):
    """Starts and runs an asynchronous task until completion.

    This function creates a new event loop, sets it as the current event loop,
    and runs the provided asynchronous task until it is complete.

    Args:
        task (coroutine function): An asynchronous function to be executed.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task())


class TalkToddyPolicy(BasePolicy, policy_name="talk_toddy"):
    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str
    connection: AsyncRealtimeConnection
    session: Session
    connected: asyncio.Event

    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32], ip: str
    ) -> None:
        """Initializes an instance of the class with specified parameters and sets up audio and communication components.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot associated with this instance.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            ip (str): IP address for the ZMQ sender node.
        """
        super().__init__(name, robot, init_motor_pos)

        mic = Microphone()
        speaker = Speaker()

        self.mic_device = mic.device
        self.speaker_device = speaker.device

        self.zmq_receiver = ZMQNode(type="receiver")
        self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.is_prepared = False
        self.is_recording = False
        self.last_control_inputs: Dict[str, float] = {}
        self.do_push_up = False
        self.is_push_up_time_set = False
        self.conversation_done = False

        self.pushup_policy = ReplayPolicy(
            "push_up", robot, init_motor_pos, run_name="push_up"
        )

        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync(self.speaker_device, 2.0)
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()

        self.thread1 = threading.Thread(
            target=start_async_task,
            args=(self.handle_realtime_connection,),
            daemon=True,
        )
        self.thread2 = threading.Thread(
            target=start_async_task, args=(self.send_mic_audio,), daemon=True
        )

        self.thread1.start()
        self.thread2.start()

    async def handle_realtime_connection(self) -> None:
        """Handle a real-time connection to the GPT-4o model.

        Establishes a connection to the GPT-4o real-time preview model and manages
        session updates and audio responses. The function listens for various event
        types such as session creation, session updates, audio data, and audio
        transcripts. It processes these events to update session information, handle
        audio playback, and track specific phrases in transcripts to trigger actions.
        """
        async with self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`
            context = TODDY_PROMPT
            await conn.session.update(
                session={
                    "model": "gpt-4o-realtime-preview",
                    "turn_detection": None,  # type: ignore
                    "instructions": context,
                    "voice": "verse",
                }  # {"type": "server_vad"}}
            )

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    # session_display: SessionDisplay = self.query_one(SessionDisplay)
                    # assert event.session.id is not None
                    # session_display.session_id = event.session.id
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

                    if "here we go" in str(acc_items[event.item_id]).lower():
                        self.do_push_up = True

                    if "hope" in str(acc_items[event.item_id]).lower():
                        self.conversation_done = True

                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        """Retrieve an active asynchronous real-time connection.

        Waits for the connection to be established and then returns the active
        `AsyncRealtimeConnection` instance.

        Returns:
            AsyncRealtimeConnection: The active real-time connection instance.

        Raises:
            AssertionError: If the connection is not established.
        """
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        """Continuously capture and send microphone audio data asynchronously.

        This function captures audio from a specified microphone device and sends it over a connection. It operates in an infinite loop, reading audio data in chunks and sending it when a condition is met. The function handles the audio stream asynchronously and ensures that audio data is sent only after a specific event is triggered.

        Raises:
            KeyboardInterrupt: If the process is interrupted by the user.
        """
        sent_audio = False

        # device_info = sd.query_devices()
        # print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            device=self.mic_device,
        )
        stream.start()

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()

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

    async def on_msg(self) -> None:
        """Handles the message event to toggle audio recording and manage session state.

        This method checks the current recording state and toggles it. If recording is active,
        it clears the `should_send_audio` event and stops recording. In manual `turn_detection`
        mode, it commits the audio buffer and initiates a response. If not recording, it sets
        the `should_send_audio` event and starts recording.
        """
        if self.is_recording:
            self.should_send_audio.clear()
            self.is_recording = False

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
            self.is_recording = True

    def step(self, obs: Obs, is_real: bool = False):
        """Executes a step in the control loop, handling preparation, push-up actions, and communication.

        Args:
            obs (Obs): The current observation containing time and other relevant data.
            is_real (bool, optional): Flag indicating whether the operation is in a real environment. Defaults to False.

        Returns:
            tuple: A dictionary of control inputs and the current action or motor position.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 12.0 if is_real else 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=10.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if self.do_push_up:
            if not self.is_push_up_time_set:
                self.is_push_up_time_set = True
                self.pushup_policy.time_start = obs.time

            if self.pushup_policy.is_done:
                self.do_push_up = False
                asyncio.run(self.on_msg())
                asyncio.run(self.on_msg())
            else:
                return self.pushup_policy.step(obs, is_real)

        send_msg = ZMQMessage(
            time=time.time(),
            control_inputs={"listen": int(self.audio_player.is_playing())},
        )
        self.zmq_sender.send_msg(send_msg)

        if not self.do_push_up and not self.conversation_done:
            control_inputs = {}
            msg = self.zmq_receiver.get_msg()
            if msg is not None and msg.control_inputs is not None:
                control_inputs = msg.control_inputs

            if (
                "listen" in control_inputs
                and "listen" in self.last_control_inputs
                and control_inputs["listen"] != self.last_control_inputs["listen"]
            ):
                print("Toggle listening...")
                asyncio.run(self.on_msg())

            if len(control_inputs) > 0 and "listen" in control_inputs:
                self.last_control_inputs = control_inputs

        return {}, self.default_motor_pos
