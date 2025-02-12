import threading
from typing import List

import numpy as np
import sounddevice as sd
import soxr

CHANNELS = 1
SAMPLE_RATE = 24000
SPEAKER_SR = 44100
# CHUNK_LENGTH_S = 0.05  # 100ms


class AudioPlayerAsync:
    """Asynchronous audio player using sounddevice."""

    def __init__(self, device: int, volume: float = 1.0):
        """Initializes an audio output stream with specified device and volume settings.

        Args:
            device (int): The identifier for the audio output device.
            volume (float, optional): The playback volume level, ranging from 0.0 (mute) to 1.0 (full volume). Defaults to 1.0.

        Attributes:
            queue (List[np.ndarray]): A list to store audio data to be played.
            lock (threading.Lock): A lock to ensure thread-safe operations on the queue.
            stream (sd.OutputStream): The audio output stream for playback.
            estimated_end_time (int): The estimated time when the current audio playback will end.
            playing (bool): A flag indicating whether audio is currently being played.
            _frame_count (int): A counter for the number of audio frames processed.

        """
        self.queue: List[np.ndarray] = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            # samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            # dtype=np.int16,
            # blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
            device=device,
        )
        self.estimated_end_time = 0
        self.volume = volume
        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        """Handles audio data processing for output by filling the buffer with queued data.

        This method is called to process and output audio data. It retrieves audio data
        from a queue, fills the output buffer, and ensures that the buffer is completely
        filled by appending zeros if necessary.

        Args:
            outdata (numpy.ndarray): The output buffer to be filled with audio data.
            frames (int): The number of frames to be processed.
            time (object): Timing information for the callback.
            status (object): Status information for the callback.

        """
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
        """Resets the frame count to zero.

        This method sets the internal frame count attribute to zero, effectively resetting any count of frames that may have been accumulated.

        """
        self._frame_count = 0

    def get_frame_count(self):
        """Retrieve the current frame count.

        Returns:
            int: The number of frames currently counted.
        """
        return self._frame_count

    def add_data(self, data: bytes):
        """Adds PCM16 single-channel audio data to the processing queue, applying volume adjustment and resampling if necessary.

        Args:
            data (bytes): PCM16 single-channel audio data to be processed.
        """
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            np_data *= self.volume
            if SAMPLE_RATE != SPEAKER_SR:
                np_data = soxr.resample(np_data, SAMPLE_RATE, SPEAKER_SR)

            self.queue.append(np_data)

            # Recalculate estimated end time dynamically
            if self.playing:
                self.estimated_end_time = self.stream.time + self.get_total_duration()
            else:
                self.start()

    def start(self):
        """Initiates the playback by setting the playing state to True and starting the stream."""
        self.playing = True
        self.stream.start()

    def stop(self):
        """Stops the playback and clears the queue.

        Sets the playing status to False, stops the audio stream, and clears the queue of any pending items.
        """
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        """Closes the stream associated with the current instance."""
        self.stream.close()

    def is_playing(self):
        """Determine if playback is currently active.

        Returns:
            bool: True if playback is active and the current stream time is less than the estimated end time plus a buffer of 0.5 seconds; otherwise, False.
        """
        return self.playing and self.stream.time < self.estimated_end_time + 0.5

    def get_total_duration(self):
        """Calculate the total duration of audio samples in the queue.

        Returns:
            float: The total duration of the queued audio samples in seconds,
            calculated by dividing the total number of samples by the speaker's
            sample rate (SPEAKER_SR).
        """
        total_samples = sum(len(chunk) for chunk in self.queue)
        return total_samples / SPEAKER_SR
