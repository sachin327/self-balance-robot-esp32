# Board inventory

Captured 2026-09-14 from `pio run -e sysinfo -t upload -t monitor`
(`src/sysinfo/main.cpp`). Re-run that target to refresh these numbers.

Board: ESP32 DevKit v1 (`esp32doit-devkit-v1`), on `/dev/ttyUSB0`.

## Verdict

The board is not the limitation. 343 KB free heap against a 12 KB TFLite arena,
112 KB largest contiguous block, and ~1 MB free in the app partition. Every
constraint on the policy model is **inference latency**, not capacity.

Two findings worth acting on are in [Flash speed](#flash-speed-is-halved) and
[Reading the budget table](#reading-the-budget-table) below.

## Chip

| | |
|---|---|
| model | ESP32 rev 3, 2 cores |
| cpu frequency | 240 MHz (XTAL 40 MHz) |
| radios | WiFi, BT, BLE (all unused) |
| sdk | v4.4.7 |
| arduino core | 2.0.17 |

## Flash

| | |
|---|---|
| size, configured | 4096 KB |
| size, physical | 4096 KB |
| speed | **40 MHz** |
| sketch (sysinfo build) | 266 KB |
| free | 1280 KB |

Configured and physical size agree, so the firmware is built for the right flash
part and nothing is unreachable.

### Flash speed is halved

40 MHz is half what this board supports. Since model weights stream from flash on
every inference, and that streaming is what blows the 5 ms control budget, raising
it is close to free performance. In `platformio.ini`:

```ini
board_build.f_flash = 80000000L
```

Expect roughly 2x on weight streaming, moving the model budget from ~33 KB to
~66 KB. Verify with MODE_VALIDATE, which reports mean inference time directly.

## Memory

| | |
|---|---|
| heap total | 367.4 KB |
| heap free | 342.9 KB |
| largest free block | 112.0 KB |
| heap low-water mark | 337.4 KB |
| psram | none |

`largest free block`, not total free heap, is what governs the TFLite arena —
`AllocateTensors()` needs one contiguous allocation and the ESP32 heap
fragments. At 112 KB against a 12 KB arena (7864 bytes actually used) there is
roughly 9x headroom. Watch this line, not `heap free`, if the arena ever grows.

No PSRAM, which costs nothing here: on the classic ESP32 it is too slow to stream
weights from anyway.

## Partitions

Default table, as flashed for the `sysinfo` build. Note the `policy` build uses
`huge_app.csv` instead, which drops `app1` to give `app0` 3 MB.

| name | type | offset | size |
|---|---|---|---|
| nvs | data | 0x009000 | 20 KB |
| otadata | data | 0x00e000 | 8 KB |
| app0 | app | 0x010000 | 1280 KB |
| app1 | app | 0x150000 | 1280 KB |
| spiffs | data | 0x290000 | 1408 KB |
| coredump | data | 0x3f0000 | 64 KB |

## Model budget

Running partition `app0`, 1280 KB; sysinfo sketch uses 266 KB, leaving 1014 KB.
Under `huge_app.csv` the policy build has ~3 MB, so **flash capacity is not a
constraint on model size**.

At the measured 40 MHz flash speed, weights stream at ~6.9 MB/s, about 0.15 ms
per KB, against a 5 ms budget at 200 Hz:

| model | inference | verdict |
|---|---|---|
| 823 KB | ~123 ms | over budget |
| 206 KB | ~31 ms | over budget |
| 100 KB | ~15 ms | over budget |
| 33 KB | ~5.0 ms | at the line |
| 20 KB | ~3.0 ms | fits |
| 6 KB | ~0.9 ms | fits |

### Reading the budget table

The table extrapolates linearly from the 823 KB measurement, and that is only
valid at the large end. The ESP32 has a 32 KB flash cache: a model that fits
inside it is re-read from cache rather than flash on each inference, so real
latency below ~32 KB is **better** than the table says. The `fits` verdicts are
therefore conservative. Treat the table as a guide and measure with
MODE_VALIDATE.

Target **~20 KB**, not 33 KB, to leave headroom for the IMU read, encoder update
and motor PI inside the same 5 ms tick.

## Consequences for the model

The current 823 KB float32 model is ~24x over budget. Accuracy is not the issue —
it matches PyTorch to 5e-06. Size is.

- **int8 quantization** is supported by TFLite Micro and gives ~4x, taking 823 KB
  to ~206 KB. Still ~6x over. Not sufficient alone.
- **fp16** is storage-only on TFLite Micro — weights are dequantized to float32
  before compute, so it gives 2x on size and costs time. Wrong direction.
- **Shrinking the architecture** is the fix. A 9 -> 64 -> 64 -> 13 MLP is about
  5.5k parameters, ~22 KB as float32, which fits without quantization at all. The
  current 823 KB implies very wide hidden layers for a 9-input, 13-output problem.

Order of work: shrink the architecture in the training repo, raise `f_flash` to
80 MHz, then apply int8 only if more margin is wanted.

When retraining, regenerate `include/test_vectors.h` against the new checkpoint —
otherwise MODE_VALIDATE scores a new model against the old model's reference
outputs and reports failures that are not real.
