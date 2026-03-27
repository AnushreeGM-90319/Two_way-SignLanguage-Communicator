import torch
from transformers import pipeline
import sounddevice as sd
import scipy.io.wavfile as wav

# 1. Load Model (It will download once - about 300MB)
print("Loading Speech Recognition AI...")
asr_pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def listen_and_transcribe():
    fs = 16000 # Wav2Vec2 requires 16kHz
    duration = 4 # Seconds
    print("Listening for 4 seconds...")
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write("temp_audio.wav", fs, audio)
    
    result = asr_pipe("temp_audio.wav")
    return result["text"]

if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Transcript: {text}")