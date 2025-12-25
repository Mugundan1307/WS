#include <M5Unified.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Your model array
#include "model_kwsp_int8.cc"  // defines g_model, g_model_len

// Audio capture settings
static const int SAMPLE_RATE    = 16000;
static const int RECORD_MS      = 1000;
static const int BUFFER_SAMPLES = SAMPLE_RATE * RECORD_MS / 1000;

// Feature vector size (13 segments)
static const int kFeatureSize = 13;

int16_t audio_buffer[BUFFER_SAMPLES];
float   feature_buffer[kFeatureSize];

// Normalization arrays
const float g_mean[kFeatureSize] = {
  -502.979828, 51.004513, 18.969086, 12.869969, 10.846627, 10.133053,
  4.070014, -0.918299, 9.598298, 2.716177, 1.262040, 1.985561, 2.782920
};

const float g_std[kFeatureSize] = {
  25.326286, 10.422313, 2.920694, 4.346381, 2.270152, 3.922652,
  3.053097, 3.541523, 4.563084, 1.918527, 1.723969, 1.871765, 1.306914
};

// TFLite Micro globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  constexpr int kTensorArenaSize = 20 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

void setupModel() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(10, 10);
    M5.Display.setTextSize(2);
    M5.Display.println("Model schema error");
    while (1) delay(1000);
  }

  // Build operator resolver (max 10 ops)
  tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  interpreter = new tflite::MicroInterpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(10, 10);
    M5.Display.setTextSize(2);
    M5.Display.println("Tensor alloc error");
    while (1) delay(1000);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
}

bool record_one_second() {
  if (!M5.Mic.isEnabled()) return false;

  size_t total = 0;
  while (total < BUFFER_SAMPLES) {
    size_t remain = BUFFER_SAMPLES - total;
    if (M5.Mic.record(audio_buffer + total, remain)) {
      total += remain;
    } else {
      delay(5);
    }
  }
  return true;
}

void computeFeaturesFromAudio() {
  for (int i = 0; i < kFeatureSize; ++i) {
    int start = (BUFFER_SAMPLES * i) / kFeatureSize;
    int end   = (BUFFER_SAMPLES * (i + 1)) / kFeatureSize;
    long sum = 0;

    for (int j = start; j < end; ++j) {
      sum += abs(audio_buffer[j]);
    }

    float mean_abs = (float)sum / (float)(end - start);
    feature_buffer[i] = (mean_abs - g_mean[i]) / g_std[i];
  }
}

void runInferenceAndDisplay() {
  for (int i = 0; i < kFeatureSize; ++i) {
    input->data.f[i] = feature_buffer[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(10, 10);
    M5.Display.println("Invoke Error");
    delay(1000);
    return;
  }

  float yes_score = output->data.f[0];
  float no_score  = output->data.f[1];
  float bg_score  = output->data.f[2];

  int idx = 0;
  float best = yes_score;

  if (no_score > best) { best = no_score; idx = 1; }
  if (bg_score > best) { best = bg_score; idx = 2; }

  const char* label = (idx == 0) ? "YES" : (idx == 1) ? "NO" : "BG";

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(10, 10);
  M5.Display.setTextSize(2);
  M5.Display.printf("Pred: %s\n", label);
  M5.Display.setCursor(10, 40);
  M5.Display.printf("Y:%.2f N:%.2f B:%.2f", yes_score, no_score, bg_score);
}

void setup() {
  auto cfg = M5.config();
  cfg.internal_mic = true;
  cfg.internal_spk = false;
  M5.begin(cfg);

  M5.Display.setTextSize(2);
  M5.Display.setCursor(10, 10);
  M5.Display.println("KW Inference Init");

  M5.Mic.begin();
  setupModel();

  delay(1000);
  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(10, 10);
  M5.Display.println("Tap to test");
  M5.Display.setCursor(10, 40);
  M5.Display.println("Say YES/NO or BG");
}

void loop() {
  M5.update();
  auto t = M5.Touch.getDetail();
  if (!t.isPressed()) {
    delay(10);
    return;
  }

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(10, 10);
  M5.Display.println("Say keyword in 0.5s...");

  delay(500);

  if (!record_one_second()) {
    M5.Display.setCursor(10, 40);
    M5.Display.setTextColor(RED);
    M5.Display.println("Record failed");
    delay(1000);
    return;
  }

  computeFeaturesFromAudio();
  runInferenceAndDisplay();

  delay(2000);
  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(10, 10);
  M5.Display.println("Tap to test");
  M5.Display.setCursor(10, 40);
  M5.Display.println("Say YES/NO or BG");
}
