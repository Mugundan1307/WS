Task 11 -  Keyword Spotting (KWS)

Objectives:
- Record labeled audio samples (Yes/No/Background) using M5Stack Core2
- Extract MFCC features using Python on your laptop
- Train a TinyML neural network model using TensorFlow
- Run inference on the laptop to classify new audio

Materials:
- M5Stack Core2
- Arduino IDE
- Anaconda/Python 3.10+
- Laptop with USB‑C port
- SD Card (for Core2)

Part 1: Setup Arduino IDE and M5Stack Core2

Step 1.1 Install Arduino IDE
1. Download Arduino IDE from https://www.arduino.cc/software
2. Install the IDE
3. Open Arduino IDE

Step 1.2 Install M5Stack Board Support
1. In Arduino IDE: Sketch → Include Library → Manage Libraries…
2. Search for `M5Stack`
3. Click **Install** on the latest version
4. Select board: Tools → Board → M5Stack → M5Stack-Core2

Step 1.3 Install M5Unified Library
1. Go to: Sketch → Include Library → Manage Libraries…
2. Search for `M5Unified`
3. Click **Install** on the latest version

Step 1.4 Connect Core2 and Select Port
1. Connect M5Stack Core2 to your device via USB‑C **data cable**
2. In Arduino IDE: **Tools → Port**
3. Select the `/dev/cu.usb*` or `/dev/cu.wchusbserial*` device
   - **Do NOT select** `/dev/cu.debug-console`
4. If no port appears, install USB drivers:
   - Silicon Labs CP210x driver
   - WCH CH9102 driver
   - Fully power off and restart your Mac after installation
5. Re-plug Core2 and check Tools → Port again

Step 1.5 Troubleshoot Upload Issues
If upload fails with "Failed to connect to ESP32":
1. Try a different USB‑C **data** cable
2. Try a different USB port on your Mac
3. While Arduino shows "Connecting…", press the RESET button on Core2 once or twice quickly
4. Try again

Part 2: Setup Python Environment

Step 2.1 Create Conda Environment
Open Terminal and run:
```
conda create -n kws_env python=3.10
conda activate kws_env
pip install numpy librosa soundfile scikit-learn tensorflow
```
Step 2.2 Create Project Folder
```
mkdir ~/kws_project
cd ~/kws_project
mkdir -p data/raw data/wav data/features
```
Your folder structure should be:
- kws_project/
  - data/raw/ (will hold .raw files from Core2)
  - data/wav/ (will hold converted .wav files)
  - data/features/ (will hold features, model, stats)


Part 3: Record Audio on M5Stack Core2

Step 3.1 Upload Recording Sketch to Core2
1. In Arduino IDE, create a new file `Task11_Record.ino`
2. Upload the sketch to Core2
3. The sketch will initialize Core2 with microphone support and SD card storage

Step 3.2 Record Samples
On the Core2 screen you'll see: "Recording ready / L:YES M:NO R:BG"

Record approximately 50–60 samples per class:

**For YES class:** 
- Tap left side 
- Wait for recording to start 
- Say "yes" clearly within 1 second

**For NO class:** 
- Tap middle 
- Wait for recording to start 
- Say "no" clearly within 1 second

**For BG (Background) class:** 
- Tap right 
- Stay silent or make background noise within 1 second

The files will be saved as:
- yes_000.raw, yes_001.raw, ..., yes_059.raw
- no_000.raw, no_001.raw, ..., no_059.raw
- bg_000.raw, bg_001.raw, ..., bg_059.raw

Step 3.3 Copy Files from Core2 to Laptop
1. Power off Core2
2. Remove SD card
3. Insert SD card into laptop (via USB adapter if needed)
4. Copy all `.raw` files to `~/kws_project/data/raw/`

Verify with:
```
ls ~/kws_project/data/raw/
```

You should see yes_*.raw, no_*.raw, bg_*.raw files.

Part 4: Convert RAW to WAV

Step 4.1 Create raw_to_wav.py Script
In `~/kws_project`, create file `raw_to_wav.py`

This script will:
- Read all `.raw` files from data/raw/
- Convert them to `.wav` format (16-bit, 16kHz, mono)
- Save converted files to data/wav/

Step 4.2 Run Conversion
```
cd ~/kws_project
conda activate kws_env
python raw_to_wav.py
```
Expected output shows conversion of each file:
```
Converted: data/raw/yes_000.raw -> data/wav/yes_000.wav
Converted: data/raw/no_000.raw -> data/wav/no_000.wav
...
```
Verify with:
```
ls data/wav/
```
You should see yes_*.wav, no_*.wav, bg_*.wav files.



Part 5: Extract MFCC Features

Step 5.1 Create extract_mfcc.py Script
In `~/kws_project`, create file `extract_mfcc.py`

This script will:
- Load all `.wav` files
- Extract MFCC (Mel-Frequency Cepstral Coefficient) features (13 dimensions)
- Take the mean across time for each audio sample
- Save features and labels to `features.npz`

Step 5.2 Run Feature Extraction
```
python extract_mfcc.py
```
Expected output:
```
Saved features: data/features/features.npz
X shape: (150, 13) y shape: (150,)
```
(The exact numbers depend on how many samples you recorded)

Part 6: Train TinyML Model

Step 6.1 Create train_kw_model.py Script
In `~/kws_project`, create file `train_kw_model.py`

This script will:
- Load the extracted features from `features.npz`
- Compute and save normalization statistics (mean, std)
- Split data into training (80%) and validation (20%) sets
- Build a 3-layer neural network (13 input → 32 hidden → 32 hidden → 3 output)
- Train the model for 40 epochs
- Save the trained Keras model as `model_kwsp.h5`
- Save normalization stats as `norm_stats.npz`


Step 6.2 Run Training
```
python train_kw_model.py
```
Expected output shows training progress:
```
Loaded X shape: (150, 13) y shape: (150,)
Training set size: 120
Validation set size: 30
Training model...
Epoch 1/40
[training progress...]
...
Validation accuracy: 0.95
Saved Keras model to data/features/model_kwsp.h5
Saved norm stats to data/features/norm_stats.npz
```

After completion, verify:
```
ls data/features/
```

You should see:
- model_kwsp.h5 (Keras model)
- norm_stats.npz (mean/std arrays)
- model_kwsp_int8.tflite (optional, for on-device deployment)

Part 7: Run Inference on Laptop

Step 7.1 Create pc_inference_wav.py Script
In `~/kws_project`, create file `pc_inference_wav.py`

This script will:
- Load the trained Keras model
- Load normalization statistics
- Accept a `.wav` file as input
- Extract MFCC features from the audio
- Normalize features using training statistics
- Run inference through the model
- Display prediction (YES/NO/BG) with confidence scores

Step 7.2 Run Inference on Test Samples

Test on your recorded samples:
```
python pc_inference_wav.py data/wav/yes_000.wav
python pc_inference_wav.py data/wav/no_000.wav
python pc_inference_wav.py data/wav/bg_000.wav
```
Expected output for each command:
```
File: data/wav/yes_000.wav
Prediction: YES
Scores: YES=0.950, NO=0.030, BG=0.020

File: data/wav/no_000.wav
Prediction: NO
Scores: YES=0.010, NO=0.980, BG=0.010

File: data/wav/bg_000.wav
Prediction: BG
Scores: YES=0.020, NO=0.030, BG=0.950
```

Expected File Structure After Completion

```
~/kws_project/
├── data/
│   ├── raw/
│   │   ├── yes_000.raw through yes_059.raw
│   │   ├── no_000.raw through no_059.raw
│   │   └── bg_000.raw through bg_059.raw
│   ├── wav/
│   │   ├── yes_000.wav through yes_059.wav
│   │   ├── no_000.wav through no_059.wav
│   │   └── bg_000.wav through bg_059.wav
│   └── features/
│       ├── features.npz
│       ├── norm_stats.npz
│       ├── model_kwsp.h5
│       └── model_kwsp_int8.tflite
├── raw_to_wav.py
├── extract_mfcc.py
├── train_kw_model.py
└── pc_inference_wav.py




