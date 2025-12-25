#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <M5Unified.h>
#include <M5GFX.h>
#define SD_SPI_CS_PIN   4
#define SD_SPI_SCK_PIN  18
#define SD_SPI_MISO_PIN 38
#define SD_SPI_MOSI_PIN 23

static const int SAMPLE_RATE    = 16000;   // Hz
static const int RECORD_MS      = 1000;    // 1 second
static const int BUFFER_SAMPLES = SAMPLE_RATE * RECORD_MS / 1000;

int16_t audio_buffer[BUFFER_SAMPLES];

int yesCount = 0;
int noCount  = 0;
int bgCount  = 0;

void drawUI() {
  M5.Display.fillScreen(BLACK);
  M5.Display.setTextSize(0.5);
  M5.Display.setCursor(10, 10);
  M5.Display.println("Keyword Record");

  M5.Display.setCursor(10, 60);
  M5.Display.println("Left:  YES");
  M5.Display.setCursor(10, 90);
  M5.Display.println("Middle: NO");
  M5.Display.setCursor(10, 120);
  M5.Display.println("Right: BG");

  M5.Display.setCursor(10, 170);
  M5.Display.println("Tap region to record 1s");
}

// record 1 second into audio_buffer
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

// save buffer as raw 16-bit PCM
bool save_raw(const char *filename) {
  File f = SD.open(filename, FILE_WRITE);
  if (!f) {
    M5.Display.setCursor(10, 210);
    M5.Display.setTextColor(RED);
    M5.Display.printf("File open error: %s\n", filename);
    return false;
  }
  f.write((uint8_t *)audio_buffer, BUFFER_SAMPLES * sizeof(int16_t));
  f.close();
  return true;
}

void setup() {
  M5.begin();

  M5.Display.setTextFont(&fonts::Orbitron_Light_24);
  M5.Display.setTextSize(0.5);

  // SD Card Initialization
  SPI.begin(SD_SPI_SCK_PIN, SD_SPI_MISO_PIN, SD_SPI_MOSI_PIN, SD_SPI_CS_PIN);
  if (!SD.begin(SD_SPI_CS_PIN, SPI, 25000000)) {
    // Print a message if SD card initialization failed or if the SD card does not exist.
    M5.Display.print("\n SD card not detected\n");
    while (1)
      ;
  } else {
    M5.Display.print("\n SD card detected\n");
  }
  delay(1000);

  // start internal mic with default settings[web:82]
  M5.Mic.begin();

  delay(2000);
  drawUI();
}

void loop() {
  M5.update();

  auto t = M5.Touch.getDetail();
  if (!t.isPressed()) {
    delay(10);
    return;
  }

  int16_t x = t.x;
  int32_t w = M5.Display.width();
  int32_t region = x * 3 / w;   // 0: left, 1: mid, 2: right

  const char *label = nullptr;
  char filename[32];

  if (region == 0) {
    label = "YES";
    sprintf(filename, "/yes_%03d.raw", yesCount++);
  } else if (region == 1) {
    label = "NO";
    sprintf(filename, "/no_%03d.raw", noCount++);
  } else {
    label = "BG";
    sprintf(filename, "/bg_%03d.raw", bgCount++);
  }

  M5.Display.fillScreen(BLACK);
  M5.Display.setTextSize(0.5);
  M5.Display.setCursor(10, 10);
  M5.Display.printf("Recording %s...\n", label);

  delay(100);

  if (!record_one_second()) {
    M5.Display.setCursor(10, 60);
    M5.Display.setTextColor(RED);
    M5.Display.println("Mic read failed");
    delay(2000);
    drawUI();
    return;
  }

  if (save_raw(filename)) {
    M5.Display.setCursor(10, 60);
    M5.Display.setTextColor(GREEN);
    M5.Display.printf("Saved: %s\n", filename);
  } else {
    M5.Display.setCursor(10, 60);
    M5.Display.setTextColor(RED);
    M5.Display.printf("Save failed: %s\n", filename);
  }

  delay(2000);
  drawUI();
}
