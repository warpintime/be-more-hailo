# AUDIO CAPTURE & WAKE WORD
import sounddevice as sd
import numpy as np
import wave
import threading
from openwakeword.model import Model
from . import config

class Ears:
    def __init__(self, callback=None):
        self.stop_event = threading.Event()
        self.model = None # Lazy load
        self.wakeword_active = False

    def load_model(self):
        try:
            # Fix for openwakeword v0.5+ vs v0.4
            try:
                self.model = Model(wakeword_model_paths=[config.WAKE_WORD_MODEL])
            except TypeError:
                self.model = Model(wakeword_models=[config.WAKE_WORD_MODEL])
        except Exception as e:
            print(f"[EARS] Failed to load OpenWakeWord: {e}")

    def listen_loop(self, on_wake, on_audio_chunk=None):
        """
        Main audio loop capturing 48kHz, downsampling to 16kHz for WakeWord,
        passing raw audio chunks for processing.
        """
        if not self.model: self.load_model()

        CHUNK = 1280 # ~80ms at 16k
        capture_rate = config.MIC_RATE
        target_rate = 16000
        ds_factor = capture_rate // target_rate

        def audio_callback(indata, frames, time, status):
            if status: print(status)
            # 1. Check Wake Word
            # Downsample: simple slice [::3] for 48->16k
            audio_16k = indata[::ds_factor, 0].flatten() # take channel 0

            if self.model:
                self.model.predict(audio_16k)
                for key in self.model.prediction_buffer.keys():
                    if self.model.prediction_buffer[key][-1] > config.WAKE_THRESHOLD:
                        self.model.reset()
                        on_wake()
                        break

            # 2. Pass data (for recording if needed)
            if on_audio_chunk:
                on_audio_chunk(indata.copy())

        with sd.InputStream(samplerate=capture_rate, device=config.MIC_INDEX,
                            channels=1, dtype='int16', callback=audio_callback, blocksize=CHUNK*ds_factor):
            self.stop_event.wait()

    def capture_audio(self, max_seconds=10):
        """Record audio until silence is detected, returning the WAV filepath."""
        filename = "input.wav"
        frames = []
        silent_chunks = 0
        has_spoken = False

        def callback(indata, frame_count, time_info, status):
            nonlocal silent_chunks, has_spoken
            vol = np.linalg.norm(indata) * 10
            frames.append(indata.copy())
            if vol < 50000:
                silent_chunks += 1
            else:
                silent_chunks = 0
                has_spoken = True

        try:
            with sd.InputStream(samplerate=config.MIC_RATE, device=config.MIC_INDEX,
                                channels=1, dtype='int16', callback=callback):
                while not self.stop_event.is_set():
                    sd.sleep(50)
                    if not has_spoken and silent_chunks > 100:
                        break
                    if has_spoken and silent_chunks > 40:
                        break
                    if len(frames) > (config.MIC_RATE * max_seconds / 512):
                        break
        except Exception as e:
            print(f"Recording Error: {e}")
            return None

        if not frames:
            return None

        data = np.concatenate(frames, axis=0)
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.MIC_RATE)
            wf.writeframes(data.tobytes())
        return filename

    def stop(self):
        self.stop_event.set()
