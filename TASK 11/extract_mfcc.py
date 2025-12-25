import os
import numpy as np
import librosa

WAV_DIR = "data/wav"
OUT_PATH = "data/features/features.npz"

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512

X = []   # features
y = []   # labels

label_map = {"yes": 0, "no": 1, "bg": 2}

for fname in os.listdir(WAV_DIR):
    if not fname.endswith(".wav"):
        continue

    # infer label from file name prefix
    label_str = fname.split("_")[0].lower()
    if label_str not in label_map:
        print("Skip (unknown label):", fname)
        continue

    label = label_map[label_str]
    wav_path = os.path.join(WAV_DIR, fname)

    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # sanity: enforce exactly 1 second
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    elif len(audio) > SAMPLE_RATE:
        audio = audio[:SAMPLE_RATE]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )  # (n_mfcc, frames)

    # average over time → 13-dim vector
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)  # shape (13,)

    X.append(mfcc_mean)
    y.append(label)

X = np.stack(X, axis=0)
y = np.array(y, dtype=np.int32)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez(OUT_PATH, X=X, y=y)

print("Saved features to", OUT_PATH)
print("X shape:", X.shape, "y shape:", y.shape)

