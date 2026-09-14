# models/

Where the `.tflite` that gets compiled into the firmware lives.

## Importing a model

Drop a `.tflite` in this folder, then from the project root:

```bash
python3 tools/import_tflite.py
```

That regenerates the two files the firmware links against:

```
models/<yours>.tflite          the file you dropped in
        │
        │  tools/import_tflite.py
        ▼
models/current.tflite          copy of exactly what is compiled in
include/model_data.h           extern declarations
src/policy/model_data.cpp      the model as an alignas(16) byte array
        │
        │  pio run -e policy -t upload
        ▼
ESP32 flash
```

Then `pio run -e policy -t upload`.

Run it with `~/.venvs/tflite_conv/bin/python` instead of `python3` and it will
also print the operator set and check the input/output shapes. Worth doing — a
model whose dimensions disagree with `robot_config.h` compiles fine and only
fails at boot.

## Starting from a PyTorch checkpoint

`convert_tflite.py` in the training repo
(`~/Desktop/sachin/github/robotics-examples/SelfBalancingRobotIMU/`) goes
`.pt → ONNX → .tflite` and writes the C++ arrays here directly, so it replaces
both steps above:

```bash
~/.venvs/tflite_conv/bin/python convert_tflite.py --ckpt models/model_ep250.pt
~/.venvs/tflite_conv/bin/python gen_test_vectors.py --ckpt models/model_ep250.pt
```

The second command regenerates `include/test_vectors.h`, the 50-observation
fixture MODE_VALIDATE scores against. **Regenerate it whenever the model
changes** — otherwise MODE_VALIDATE compares a new model against an old model's
reference outputs and reports failures that are not real.

## Contract the firmware expects

Set in `include/robot_config.h`, from `train_obs9_baseline.py`:

| | |
|---|---|
| input | `(1, 9)` float32, each element normalised and clamped to [-1, 1] |
| output | `(1, 13)` float32 Q-values; `argmax` indexes `ACTIONS` |
| operators | ADD, DIV, FULLY_CONNECTED, MEAN, MUL, RELU, SQRT, SUB |
| MAX_VEL | 5.7 rad/s = 0.171 m/s at the 30 mm wheel |

The eight operators must all be registered in the `MicroMutableOpResolver` in
`src/policy/policy.cpp`. LayerNorm has no TFLite operator of its own — MEAN, SUB,
MUL, ADD, SQRT and DIV are what it decomposes into.

## Size and speed

Weights stream from flash on every inference at roughly **6.9 MB/s**, measured on
this board. That works out to about **0.15 ms per KB of model**, and the 200 Hz
control loop allows 5 ms total.

The current 804 KB float32 model therefore takes ~120 ms per inference, which is
why MODE_RUN cannot balance the robot yet. Accuracy is not the problem — it
matches PyTorch to 5e-06. Size is. Roughly 33 KB is the budget for 5 ms.
