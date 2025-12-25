import os
import numpy as np
import wave

RAW_DIR = "data/raw"
WAV_DIR = "data/wav"
os.makedirs(WAV_DIR, exist_ok=True)

SAMPLE_RATE = 16000   # matches your Arduino code
SAMPLE_WIDTH = 2      # bytes (16-bit)
CHANNELS = 1

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".raw"):
        continue

    raw_path = os.path.join(RAW_DIR, fname)
    wav_name = os.path.splitext(fname)[0] + ".wav"
    wav_path = os.path.join(WAV_DIR, wav_name)

    # read as 16-bit little-endian PCM
    data = np.fromfile(raw_path, dtype=np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())

    print("Converted:", raw_path, "->", wav_path)

