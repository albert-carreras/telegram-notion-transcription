import os
import sys
import threading
import wave
from contextlib import contextmanager

import numpy as np
import ollama
import pyaudio
import sounddevice as sd
import whisperx
from TTS.api import TTS
from scipy.io.wavfile import write

sample_rate = 44100
channels = 1
device = "cpu"
audio_file = "input.wav"
output_file = "output.wav"
batch_size = 1
compute_type = "int8"
tts_model = "tts_models/en/ljspeech/tacotron2-DDC"


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

            sys.stderr = old_stderr


class ConversationManager:
    def __init__(self):
        self.history = [{'role': 'system', 'content': """You are a friendly person that answers in short sentences.
        You are having a casual conversation and you're interested in the person talking to you."""}]

    def add_user_message(self, message):
        self.history.append({'role': 'user', 'content': message})

    def add_assistant_message(self, message):
        self.history.append({'role': 'assistant', 'content': message})

    def get_history(self):
        return self.history


def record_audio(sample_rate, channels):
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    print("\n--- Recording started. Press Enter to stop recording. ---")
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        input()

    return np.concatenate(recording)


def play_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    with suppress_stdout_stderr():
        tts = TTS(tts_model).to(device)
        model = whisperx.load_model("base", device, compute_type=compute_type, language='en')
    conversation_manager = ConversationManager()

    while True:
        try:
            recording = record_audio(sample_rate, channels)
            write(audio_file, sample_rate, recording)
            audio = whisperx.load_audio(audio_file)
            print("--- WhisperX Transcribing ---")
            with suppress_stdout_stderr():
                result = model.transcribe(audio, batch_size=batch_size)

            user_input = result["segments"][0]['text']
            conversation_manager.add_user_message(user_input)

            print("--- LLM Generating ---")
            response_thread = threading.Thread(target=generate_response, args=(conversation_manager, tts))
            response_thread.start()
            response_thread.join()

            print("\n--- Press Enter to continue the conversation or type 'q' to exit. ---")
            user_choice = input().strip().lower()
            if user_choice == 'q':
                break
        except Exception as e:
            print(f"An error occurred: {str(e)}")


def generate_response(conversation_manager, tts):
    try:
        response = ollama.chat(
            model='llama3',
            messages=conversation_manager.get_history(),
        )

        assistant_response = response['message']['content']
        conversation_manager.add_assistant_message(assistant_response)

        print("--- TTS Generating ---")
        with suppress_stdout_stderr():
            tts.tts_to_file(text=assistant_response, file_path=output_file, split_sentences=False, speed=1.4)
        print("--- Response Playback ---")
        play_audio(output_file)
    except Exception as e:
        print(f"An error occurred during response generation: {str(e)}")


if __name__ == '__main__':
    main()
