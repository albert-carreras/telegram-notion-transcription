import os
import threading
from contextlib import contextmanager

import numpy as np
import ollama
import sounddevice as sd
import whisperx
from scipy.io.wavfile import write
from styletts2 import tts

sample_rate = 24000
channels = 1
device = "cpu"
audio_file = "input.wav"
batch_size = 1
compute_type = "int8"


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


with suppress_stdout_stderr():
    whisper_model = whisperx.load_model(
        "base", device, compute_type=compute_type, language="en"
    )

global styletts
styletts = None


def load_styletts():
    global styletts
    with suppress_stdout_stderr():
        styletts = tts.StyleTTS2()


class ConversationManager:
    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": """You are a friendly person that answers in short sentences.
        You are having a casual conversation and you're interested in the person talking to you.""",
            }
        ]

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_history(self):
        return self.history


def record_audio():
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print("Recording Error:", status)
        recording.append(indata.copy())

    print("\n--- Recording started. Press Enter to stop recording. ---")
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=channels, callback=callback
        ):
            input()
    except Exception as e:
        print("An error occurred during recording:", e)

    return np.concatenate(recording) if recording else np.zeros((sample_rate, channels))


def play_wav_array(wav_array):
    sd.play(wav_array, samplerate=sample_rate, blocking=True)
    sd.wait()


def generate_response(conversation_manager):
    try:
        response = ollama.chat(
            model="llama3",
            messages=conversation_manager.get_history(),
        )

        assistant_response = response["message"]["content"]
        conversation_manager.add_assistant_message(assistant_response)

        print("--- TTS Generating ---")
        if styletts is None:
            print("StyleTTS not ready yet.")
            return
        with suppress_stdout_stderr():
            wav = styletts.inference(assistant_response, diffusion_steps=4)
        print("--- Response Playback ---")
        play_wav_array(wav)
    except Exception as e:
        print(f"An error occurred during response generation: {str(e)}")


def main():
    conversation_manager = ConversationManager()
    styletts_thread = threading.Thread(target=load_styletts)
    styletts_thread.start()

    while True:
        try:
            recording = record_audio()

            if not np.any(recording):
                print("No audio recorded, please try again.")
                continue
            write(audio_file, sample_rate, recording)
            audio = whisperx.load_audio(audio_file)
            print("--- WhisperX Transcribing ---")
            if whisper_model is None:
                print("WhisperX not ready yet, please wait...")
                return
            with suppress_stdout_stderr():
                result = whisper_model.transcribe(audio, batch_size=batch_size)

            user_input = (
                result["segments"][0]["text"]
                if result["segments"]
                else "No speech detected."
            )
            conversation_manager.add_user_message(user_input)

            print("--- LLM Generating ---")
            generate_response(conversation_manager)

            print(
                "\n--- Press Enter to continue the conversation or type 'q' to exit ---"
            )
            user_choice = input().strip().lower()
            if user_choice == "q":
                print("Exiting conversation")
                break
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
