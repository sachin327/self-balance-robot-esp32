// main.cpp — board inventory. Flash it, open the monitor, read the numbers.
//
//   pio run -e sysinfo -t upload && pio device monitor
//
// Written for sizing the policy model. The two numbers that decide whether a
// model fits are printed under BUDGET at the bottom:
//
//   app partition free   how much .tflite can be linked in at all
//   largest free block   how much arena TFLite Micro can actually allocate
//
// Note the difference between this and `esptool.py flash_id`. esptool reads the
// physical chip; ESP.getFlashChipSize() below reports what the bootloader header
// was *configured* with. If the two disagree, esptool is right — the firmware is
// built for the wrong flash size and anything above the configured size is
// unreachable.

#include <Arduino.h>
#include <esp_chip_info.h>
#include <esp_ota_ops.h>
#include <esp_partition.h>
#include <esp_system.h>

#if __has_include(<esp_flash.h>)
#include <esp_flash.h>
#define HAVE_ESP_FLASH 1
#endif

namespace {

void rule(const char* title) {
  Serial.printf("\n== %s ", title);
  for (size_t i = strlen(title); i < 60; i++) Serial.print('=');
  Serial.println();
}

// Bytes are unreadable at this scale; KB with one decimal is not.
void kb(const char* label, uint32_t bytes) {
  Serial.printf("  %-24s %8.1f KB  (%lu bytes)\n",
                label, bytes / 1024.0, (unsigned long)bytes);
}

const char* chipModel(esp_chip_model_t m) {
  switch (m) {
    case CHIP_ESP32:   return "ESP32";
    case CHIP_ESP32S2: return "ESP32-S2";
    case CHIP_ESP32S3: return "ESP32-S3";
    case CHIP_ESP32C3: return "ESP32-C3";
    default:           return "unknown";
  }
}

const char* resetReason(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:  return "power-on";
    case ESP_RST_EXT:      return "external pin";
    case ESP_RST_SW:       return "software";
    case ESP_RST_PANIC:    return "panic / exception";
    case ESP_RST_INT_WDT:  return "interrupt watchdog";
    case ESP_RST_TASK_WDT: return "task watchdog";
    case ESP_RST_WDT:      return "other watchdog";
    case ESP_RST_BROWNOUT: return "brownout";
    default:               return "unknown";
  }
}

void printChip() {
  rule("CHIP");
  esp_chip_info_t info;
  esp_chip_info(&info);

  Serial.printf("  %-24s %s rev %d, %d core%s\n", "model", chipModel(info.model),
                info.revision, info.cores, info.cores == 1 ? "" : "s");
  Serial.printf("  %-24s %lu MHz (XTAL %lu MHz)\n", "cpu frequency",
                (unsigned long)getCpuFrequencyMhz(),
                (unsigned long)getXtalFrequencyMhz());
  Serial.printf("  %-24s %s%s%s\n", "radios",
                (info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi " : "",
                (info.features & CHIP_FEATURE_BT) ? "BT " : "",
                (info.features & CHIP_FEATURE_BLE) ? "BLE" : "");
  Serial.printf("  %-24s %s\n", "last reset",
                resetReason(esp_reset_reason()));
  Serial.printf("  %-24s %s\n", "sdk", ESP.getSdkVersion());
  Serial.printf("  %-24s %d.%d.%d\n", "arduino core",
                ESP_ARDUINO_VERSION_MAJOR, ESP_ARDUINO_VERSION_MINOR,
                ESP_ARDUINO_VERSION_PATCH);
}

void printFlash() {
  rule("FLASH");

  // Configured size, from the bootloader header.
  kb("size (configured)", ESP.getFlashChipSize());
  Serial.printf("  %-24s %lu MHz\n", "speed",
                (unsigned long)(ESP.getFlashChipSpeed() / 1000000));

#ifdef HAVE_ESP_FLASH
  // Physical size, read from the chip itself. Should match the line above.
  uint32_t phys = 0;
  if (esp_flash_get_physical_size(NULL, &phys) == ESP_OK) {
    kb("size (physical)", phys);
    if (phys != ESP.getFlashChipSize()) {
      Serial.println("  ** MISMATCH — firmware is built for the wrong flash "
                     "size; the excess is unreachable.");
    }
  }
#endif

  kb("sketch uses", ESP.getSketchSize());
  kb("sketch free", ESP.getFreeSketchSpace());
}

void printMemory() {
  rule("MEMORY");

  kb("heap total", ESP.getHeapSize());
  kb("heap free", ESP.getFreeHeap());
  // A TFLite arena is one contiguous allocation, so this — not total free —
  // is the number that governs whether AllocateTensors() succeeds.
  kb("largest free block", ESP.getMaxAllocHeap());
  kb("heap low-water mark", ESP.getMinFreeHeap());

  if (ESP.getPsramSize() > 0) {
    kb("psram total", ESP.getPsramSize());
    kb("psram free", ESP.getFreePsram());
  } else {
    Serial.printf("  %-24s none detected\n", "psram");
  }
}

void printPartitions() {
  rule("PARTITIONS");
  Serial.printf("  %-10s %-8s %-10s %10s %10s\n",
                "name", "type", "subtype", "offset", "size");

  esp_partition_iterator_t it =
      esp_partition_find(ESP_PARTITION_TYPE_ANY, ESP_PARTITION_SUBTYPE_ANY, NULL);
  while (it != NULL) {
    const esp_partition_t* p = esp_partition_get(it);
    Serial.printf("  %-10s %-8s 0x%-8x 0x%08lx %7.0f KB\n", p->label,
                  p->type == ESP_PARTITION_TYPE_APP ? "app" : "data",
                  p->subtype, (unsigned long)p->address, p->size / 1024.0);
    it = esp_partition_next(it);
  }
  esp_partition_iterator_release(it);
}

// The whole reason this target exists.
void printBudget() {
  rule("BUDGET — what a .tflite model has to fit in");

  const esp_partition_t* app = esp_ota_get_running_partition();
  if (app != NULL) {
    const uint32_t used = ESP.getSketchSize();
    Serial.printf("  running partition '%s', %.0f KB\n", app->label,
                  app->size / 1024.0);
    kb("  this sketch", used);
    kb("  free for a model", app->size - used);
    Serial.println("  (the policy build is larger than this one — treat the");
    Serial.println("   figure as an upper bound, not a target.)");
  }

  Serial.println();
  kb("largest contiguous heap", ESP.getMaxAllocHeap());
  Serial.println("  the TFLite arena must fit in one piece here.");

  // Measured on this board: weights stream from flash on every inference at
  // ~6.9 MB/s, i.e. ~0.15 ms per KB. The 200 Hz control loop allows 5 ms.
  Serial.println();
  Serial.println("  flash read ~6.9 MB/s  ->  ~0.15 ms per KB of model");
  Serial.println("  200 Hz control loop   ->  5.0 ms per tick");
  Serial.println();
  Serial.printf("  %-12s %-14s %s\n", "model", "inference", "verdict");
  const uint32_t sizes_kb[] = {823, 206, 100, 33, 20, 6};
  for (size_t i = 0; i < sizeof(sizes_kb) / sizeof(sizes_kb[0]); i++) {
    const float ms = sizes_kb[i] * 0.15f;
    Serial.printf("  %6lu KB    %7.1f ms     %s\n", (unsigned long)sizes_kb[i],
                  ms, ms < 5.0f ? "fits" : "OVER BUDGET");
  }
  Serial.println();
  Serial.println("  Flash space is not the constraint — speed is. Roughly");
  Serial.println("  20 KB leaves headroom for the IMU, encoders and motor PI.");
}

}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  delay(200);

  Serial.println("\n\nESP32 board inventory");
  printChip();
  printFlash();
  printMemory();
  printPartitions();
  printBudget();
  Serial.println("\nSend any character to re-read the live memory figures.");
}

void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read();
    printMemory();
  }
  delay(50);
}
