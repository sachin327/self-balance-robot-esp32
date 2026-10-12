#!/usr/bin/env python3
"""
validate_on_laptop.py — run the same 50-vector model check as the board's
policy_model_validate build, but on the laptop.

    ~/.venvs/tflite_conv/bin/python tools/validate_on_laptop.py

It reads the exact files the firmware builds from:

    include/test_vectors.h        the 50 inputs and PyTorch's reference outputs
    src/policy/model_data.cpp     the model bytes that get flashed

and prints the same table and summary the board prints, so the two can be read
side by side.

The model outputs one number in [-1, 1]; wheel command = output * 5.7 rad/s.
Errors are reported in rad/s of wheel command, against the PyTorch reference.

How to read the result:
  laptop error ~= board error   the board runs the model correctly; what error
                                there is came from int8 conversion
  board error  >  laptop error  the conversion is fine, something on the board
                                is wrong (unsupported kernel feature, etc.)

By default it uses TFLite's REFERENCE kernels, the same kind TFLite Micro uses on
the ESP32, so differences are not caused by the laptop using optimized kernels.
"""

import argparse
import os
import re
import sys

import numpy as np

TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
VECTORS_H   = os.path.join(PROJECT_DIR, "include", "test_vectors.h")
MODEL_CPP   = os.path.join(PROJECT_DIR, "src", "policy", "model_data.cpp")

MAX_VEL = 5.7          # rad/s, must match include/robot_config.h
MAX_ERR_LIMIT = 0.35   # rad/s, same pass limits as the firmware
MEAN_ERR_LIMIT = 0.10


def read_float_array(text: str, name: str) -> np.ndarray:
    body = text.split(name, 1)[1].split("};", 1)[0]
    rows = re.findall(r"\{([^{}]*)\}", body)
    return np.array([[float(v.strip().rstrip("f")) for v in r.split(",") if v.strip()]
                     for r in rows], dtype=np.float32)


def load_vectors(path: str):
    text = open(path).read()
    x = read_float_array(text, "kTestInputs")
    y = read_float_array(text, "kRefOutputs")
    src = re.search(r"Reference outputs are PyTorch \(([^)]*)\)", text)
    return x, y, (src.group(1) if src else "unknown")


def load_model_bytes(path: str) -> bytes:
    text = open(path).read()
    body = text.split("g_model[]", 1)[1].split("};", 1)[0]
    data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]{2})", body))
    declared = re.search(r"g_model_len\s*=\s*(\d+)", text)
    if declared and int(declared.group(1)) != len(data):
        sys.exit(f"error: model_data.cpp declares {declared.group(1)} bytes "
                 f"but the array holds {len(data)}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", help="check this .tflite instead of src/policy/model_data.cpp")
    ap.add_argument("--optimized", action="store_true",
                    help="use TFLite's optimized kernels instead of reference kernels")
    args = ap.parse_args()

    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("error: tensorflow not importable — run with "
                 "~/.venvs/tflite_conv/bin/python")

    x, y_ref, ref_src = load_vectors(VECTORS_H)
    if args.tflite:
        model = open(args.tflite, "rb").read()
        model_src = args.tflite
    else:
        model = load_model_bytes(MODEL_CPP)
        model_src = "src/policy/model_data.cpp"

    kernels = (tf.lite.experimental.OpResolverType.AUTO if args.optimized
               else tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp = tf.lite.Interpreter(model_content=model,
                                 experimental_op_resolver_type=kernels)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    print("=" * 65)
    print(" LAPTOP VALIDATION")
    print(f" model      {model_src}  ({len(model):,} bytes)")
    print(f" reference  include/test_vectors.h  (PyTorch {ref_src})")
    print(f" kernels    {'optimized' if args.optimized else 'reference (same kind as ESP32)'}")
    print("=" * 65)

    print("\n  wheel command in rad/s (network output x MAX_VEL)")
    print("  #     laptop        ref     |err|")
    errs = []
    for i in range(len(x)):
        interp.set_tensor(in_idx, x[i:i + 1])
        interp.invoke()
        got = float(np.clip(interp.get_tensor(out_idx)[0, 0], -1.0, 1.0)) * MAX_VEL
        ref = float(y_ref[i, 0]) * MAX_VEL
        errs.append(abs(got - ref))
        print(f"  {i:2d}   {got:+8.4f}   {ref:+8.4f}   {errs[-1]:7.4f}")

    mean_err, max_err = float(np.mean(errs)), float(np.max(errs))
    ok = max_err <= MAX_ERR_LIMIT and mean_err <= MEAN_ERR_LIMIT
    print("\n  ===================== SUMMARY =====================")
    print(f"  mean |error|            {mean_err:.4f} rad/s  (limit {MEAN_ERR_LIMIT:.2f})")
    print(f"  max  |error|            {max_err:.4f} rad/s  (limit {MAX_ERR_LIMIT:.2f})")
    print(f"  model vs reference      -> {'PASS' if ok else 'FAIL'}")
    print("  ===================================================")
    print("\n  Compare both error lines with the board's policy_model_validate output.")
    print("  Inference time is not measured here — laptop timing says nothing about the ESP32.")


if __name__ == "__main__":
    main()
