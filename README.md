# Self-Balancing Robot

ESP32 firmware for a self-balancing robot driven by a neural-network policy.
The policy is trained with PPO in a separate repo (`SelfBalancingRobotIMU`) and
runs here with TensorFlow Lite Micro. The model linked in now is
`model_h64_it0250` (PPO, second run of 2026-10-11: 10 inputs, position from
encoder counts, position integral, 0.3 m distance limit).

## What the model must look like

The firmware and the model have to agree on every item below. The training
repo's `README.md` section 7 is the other half of this list; change either
side only together with the other.

| | |
|---|---|
| Input | 10 float32 values, each divided by its limit and clamped to [-1, 1] |
| Output | 1 float32 in [-1, 1] (clipped inside the model) |
| Wheel command | output × 5.7 rad/s, the same for both wheels |
| Delay | the command is applied 2 control steps (10 ms) after it is computed |
| Control rate | 200 Hz (5 ms per step) |
| Format | int8 weights, one scale per weight matrix, float32 input and output |
| Operators | FullyConnected, Tanh, Quantize, Dequantize, and nothing else |
| Size | 10 → 64 → 64 → 1, 7,800 bytes |

The 10 inputs, in order:

| # | Input | Divided by |
|---|---|---|
| 0 | pitch (tilt forward/back) | 0.3 rad |
| 1 | pitch rate | 10 rad/s |
| 2 | roll (tilt sideways) | 0.3 rad |
| 3 | roll rate | 10 rad/s |
| 4 | left wheel speed | 5.7 rad/s |
| 5 | right wheel speed | 5.7 rad/s |
| 6 | forward speed estimate, `(-0.5·(w1+w2) + pitch rate) × 0.030 m` | 0.19 m/s |
| 7 | distance from where it started balancing, from the encoder counts: `0.030 m × (−mean wheel angle since start + (pitch − pitch at start))` | 0.3 m |
| 8 | the command on the wheels right now | 5.7 rad/s |
| 9 | position memory: running sum of input 7's distance × 5 ms, limited to ±0.5 m·s | 0.5 m·s |

Inputs 7 and 9 restart from zero every time the robot starts balancing. Input 9
is what lets the policy come back to its start point even though its tilt reading
is slightly off.

**Distance cut-off.** If the distance reading stays at 0.3 m for half a second —
the robot was lifted with the wheels spinning, or shoved out of range — the
motors switch off and it prints `TOO FAR`. It then restarts by itself, like after
a fall, once it is upright and still, and that spot becomes the new start point.
To move the robot, lay it down first (past 40° the motors switch off); if you lift it
upright the wheels will spin for a few seconds before the cut-off acts.

Wheel speeds are measured over a 20 ms window and rounded to steps of
0.0739 rad/s, as in training. All of these constants live in
[include/robot_config.h](include/robot_config.h) under "POLICY CONTRACT".

Models with a different input count do not run on this firmware; it refuses them
at boot ("model wants N inputs, firmware builds 10"). That includes
`model_h64_it0300` (9 inputs, leaky position, 0.5 m limit); the files changed for
the 10-input model are backed up in `backup_before_10input_*.tar.gz`.
Older models (DQN, 13 fixed wheel speeds) are refused as well. The firmware that ran them is kept in
`git stash` ("pre-PPO firmware"), and that model in
`models/dqn13_qat8_int8_prev.tflite`.

## Build & flash

`pio` is not on PATH by default — use the full path, or add it once (see
[bottom](#put-pio-on-your-path)).

```bash
cd ~/Documents/PlatformIO/Projects/SelfBalancingRobot
~/.platformio/penv/bin/pio run -e <environment> -t upload -t monitor
```

`<environment>` is one of: `policy_quick_check`, `policy_model_validate`,
`policy`, `motortest`, `sysinfo` — see the table below for what each one does.

Ctrl+C exits the monitor.

### Environments

| Command | What it flashes | Motors |
|---|---|---|
| `pio run -e policy_quick_check` | Sensor/wiring check — prints live IMU + encoder + observation values | off |
| `pio run -e policy_model_validate` | Model check — compares the linked model against 50 fixed PyTorch reference vectors, prints real measured inference time | off |
| `pio run -e policy` | **The real firmware** — closed loop, observe → infer → drive wheels | **on** |
| `pio run -e motortest` | Open-loop motor + encoder bench test | on |
| `pio run -e sysinfo` | Board inventory: chip, flash, heap, model size budget | n/a |

The three `policy*` targets are the same source code, just built three ways —
swap `-e`, nothing else to edit.

**Order of use on any new hardware change:** `policy_quick_check` first to
check sensors, then `policy_model_validate` after importing any new model, only
then `policy`.

### How `policy` starts

- **Power on:** loads the saved gyro calibration from flash. It calibrates by
  itself only when nothing valid is saved: the very first boot, or the first
  boot after a firmware change to how the sensor is read. Then put the robot
  down and let go for ~4 s; it retries until it has been still for 2 s.
- **Auto-start:** motors arm by themselves once the robot has been held
  **still and within ±5° of upright for 1 s**. No key press. Outside ±5° it
  stays off and prints `waiting: tilt ...`; the 1 s count restarts when it's
  back inside. If you power it on while moving it, it waits until it's still.
- **Fall:** past ~40° tilt the motors cut. Stand it back up straight and it
  restarts 1 s later.
- **Recalibrate:** send `c` over serial with the robot still. It saves the new
  calibration and auto-starts again 1 s later.

### One-time: IMU mount offset

If the MPU6050 board isn't perfectly square to the chassis, the policy balances
the *reading* at zero while the chassis actually leans, and a leaning robot has
to keep driving one way. Measure it once (saved to flash, survives re-flashing):

1. Flash `policy_quick_check` and open the monitor.
2. Hold the robot so the **chassis is exactly vertical**. Check with a phone
   level against its flat side, front-to-back and side-to-side.
3. Press `o` and hold still for ~5 s. It prints the saved offset, or
   `REJECTED` if the robot moved (just try again). `O` clears it.

The boot log shows the offset in use, or `not measured`.

In `policy_quick_check`, `c` also recalibrates and saves; encoder counts are
zeroed with `e`.

The one you'll type most:

```bash
~/.platformio/penv/bin/pio run -e policy_quick_check -t upload -t monitor
```

## Import a new model

The build reads exactly these files, always these paths — nothing in `models/`
is ever read automatically:

```
include/model_data.h          the model, declarations
src/policy/model_data.cpp     the model, bytes
include/test_vectors.h        50 test inputs + the PyTorch answers for THIS model
```

The training repo produces all three in `converted_models/`, named
`<model>_int8_model_data.h`, `<model>_int8_model_data.cpp` and
`<model>_test_vectors.h`. Copy them to the paths above, and change the
`#include` line at the top of the `.cpp` to `#include "model_data.h"`.
`test_vectors.h` must come from the same checkpoint as the model, or the
validation numbers mean nothing.

### Option A — copy the files yourself

If you already have (or generate elsewhere) a `model_data.h` / `model_data.cpp`
pair, just copy them in, renamed to exactly these names:

```bash
cp yourfile.h   include/model_data.h
cp yourfile.cpp src/policy/model_data.cpp
```

`model_data.h` must declare:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>

extern const unsigned char g_model[];
extern const size_t g_model_len;
```

`model_data.cpp` must define both, with `alignas(16)` on the array — TFLite
Micro reads the flatbuffer in place and traps on a misaligned pointer:

```cpp
#include "model_data.h"

alignas(16) const unsigned char g_model[] = { /* your bytes */ };
const size_t g_model_len = /* count, must match the array */;
```

### Option B — let the import script generate them

Drop a `.tflite` into `models/`, then:

```bash
python3 tools/import_tflite.py models/<name>.tflite
```

Regenerates the same two files. Run with `~/.venvs/tflite_conv/bin/python`
instead of the system `python3` if you want it to also check the model's op set
and input/output shapes against `robot_config.h` — worth doing, since a
mismatched model compiles fine and only fails at boot.

### Either way

After swapping the model, always run
`pio run -e policy_model_validate -t upload -t monitor` before `policy`. It
prints, for each of the 50 test inputs, the wheel command the board computes
next to the PyTorch answer, then:

- `mean |error|` and `max |error|` in rad/s, and PASS or FAIL
  (limits: mean 0.10, max 0.35 rad/s; one output step of the model is 0.045)
- the measured inference time against the 5 ms budget
- how that time splits across the model's operations

## Check the model on the laptop

Runs the same 50-vector check as `policy_model_validate`, on the laptop, using
the same two files the board builds from (`include/test_vectors.h` and
`src/policy/model_data.cpp`):

```bash
~/.venvs/tflite_conv/bin/python tools/validate_on_laptop.py
```

Compare its `mean |error|` and `max |error|` lines with the board's:

| Result | Meaning |
|---|---|
| laptop ≈ board | Board runs the model correctly. What error there is came from int8 conversion |
| board error > laptop error | Conversion is fine, something on the board is wrong |

The error is the difference in wheel command (rad/s) between the int8 model and
the original PyTorch model, for the same input. For `model_h64_it0250` the
laptop gives mean 0.030 and max 0.089 rad/s.

`include/test_vectors.h` must come from the **same checkpoint** as the model,
otherwise both scores are meaningless.

Options: `--tflite <file>` checks a `.tflite` instead of `model_data.cpp`;
`--optimized` uses TFLite's fast kernels instead of the reference kernels the
ESP32 uses.

## Debug prints

Most `Serial` output in the policy firmware goes through a `DEBUG_PRINTS` flag
instead of printing unconditionally, because `Serial.printf` is slow enough on
this board (tens of us to low ms depending on the USB buffer) to eat into the
real firmware's 5 ms control budget on its own — printing about a slow tick can
be what makes it slow.

The flag is set automatically per environment, nothing to pass by hand:

| Environment | `DEBUG_PRINTS` | Why |
|---|---|---|
| `policy_quick_check` | on | Nothing time-critical; the prints are the point |
| `policy_model_validate` | on | Same — motors off, no control loop running |
| `policy` | **off** | The 5 ms loop can't afford it |

Failures and safety events (FALLEN, WATCHDOG, a failed model boot or `Invoke()`)
always print, in every environment — those aren't debug noise, they're why the
motors just stopped. Only progress/diagnostic prints (model info at boot, the
periodic status line, slow-inference warnings) are gated.

To force prints on in `policy` for bench debugging, without touching
`platformio.ini`:

```bash
PLATFORMIO_BUILD_FLAGS="-DDEBUG_PRINTS=1" ~/.platformio/penv/bin/pio run -e policy -t upload -t monitor
```

(Same mechanism the other way — `-DDEBUG_PRINTS=0` — quiets `policy_quick_check`/`policy_model_validate` if ever needed.)

## Put `pio` on your PATH

```bash
echo 'export PATH="$HOME/.platformio/penv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

After that, drop the `~/.platformio/penv/bin/` prefix from every command above.

## Docs

- [docs/board_inventory.md](docs/board_inventory.md) — measured chip/flash/heap
  numbers and the model size budget they imply.
