import os
import numpy as np
import librosa
import tensorflow as tf

# paths
MODEL_PATH = "data/features/model_kwsp.h5"
NORM_PATH  = "data/features/norm_stats.npz"

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512

label_index_to_name = {0: "YES", 1: "NO", 2: "BG"}

# load model and norm stats
model = tf.keras.models.load_model(MODEL_PATH)
norm = np.load(NORM_PATH)
mean = norm["mean"]
std  = norm["std"]

def extract_features_from_wav(path):
  audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

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
  )  # (13, frames)

  mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)  # (13,)
  return mfcc_mean

def classify_wav(path):
  x = extract_features_from_wav(path)          # (13,)
  x_norm = (x - mean) / (std + 1e-6)          # normalize like training
  x_norm = np.expand_dims(x_norm, axis=0)     # (1, 13)

  probs = model.predict(x_norm, verbose=0)[0] # (3,)
  idx = int(np.argmax(probs))
  label = label_index_to_name.get(idx, "?")

  print(f"File: {path}")
  print(f"Pred: {label}")
  print(f"Scores: YES={probs[0]:.3f}, NO={probs[1]:.3f}, BG={probs[2]:.3f}")

if __name__ == "__main__":
  import sys
  if len(sys.argv) != 2:
    print("Usage: python pc_inference_wav.py path/to/file.wav")
    raise SystemExit(1)

  wav_path = sys.argv[1]
  if not os.path.exists(wav_path):
    print("File not found:", wav_path)
    raise SystemExit(1)

  classify_wav(wav_path)

