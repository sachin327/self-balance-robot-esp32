#!/usr/bin/env python3
"""
Evaluate a policy by running it in closed loop, with the SAME sensor pipeline it
was trained on — IMU at its real mounting height, 200 Hz physics and control,
the shared ComplementaryFilter, the action delay, and the exact 9-input
observation train.py builds. No exploration noise; optional push disturbances.

EVAL_MODEL selects what drives the robot:
  <name>.pt      a train.py (PPO) or pretrain_from_teacher.py checkpoint (models/)
  <name>.onnx    a convert_tflite.py export (converted_models/), via onnxruntime
  <name>.tflite  a convert_tflite.py export, fp32/fp16/int8/int16, via
                 tf.lite.Interpreter (integer I/O is (de)quantised automatically)
  pd             no model: the hand-written PD controller, on the same inputs.
                 The environment check — it should score ~100%.

Every model outputs ONE number in [-1, 1]; the wheel command is that x MAX_VEL.
All formats sit behind one policy.command(obs) adapter, so the sim loop never
knows which runtime is underneath. Testing a .tflite against its source .pt this
way is a real sim-in-the-loop check.

(The file name is historical.)

Every config value below MUST match train.py.
"""

import os

from isaaclab.app import AppLauncher

HEADLESS = os.environ.get("EVAL_HEADLESS", "0") == "1"   # watch it balance; EVAL_HEADLESS=1 for a fast headless score
app_launcher = AppLauncher(headless=HEADLESS, visualizer=None if HEADLESS else "kit")
simulation_app = app_launcher.app

import sys
import math
import random

import torch
import warp as wp
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Imu, ImuCfg

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from model import load_actor          # noqa: E402
from imu_filter import ComplementaryFilter, GYRO_BIAS_MAX, ARM_DELAY_S  # noqa: E402

# =====================================================================
# CONFIG — mirrors train.py
# =====================================================================
USD_PATH   = os.path.join(_script_dir, "data", "SelfBalancingRobotN20.usd")
# Which policy to run (see the module docstring). Pass it explicitly. The
# default is the training run's "best", which means longest episodes DURING
# training (with exploration noise) — a starting guess, not a verdict. Score the
# numbered itNNNN checkpoints too and rank them by success rate here.
_EVAL_MODEL = os.environ.get("EVAL_MODEL", "model_h64_best.pt")


def _resolve_model_path(name: str) -> str:
    """
    Accept a .pt checkpoint, or a .onnx/.tflite produced by convert_tflite.py.

    A bare filename is looked up in the conventional directory for its
    extension (models/ for .pt, converted_models/ for .onnx and .tflite),
    matching where train.py and convert_tflite.py actually write them. A path
    that already contains a directory separator, or that exists as given
    (relative to cwd), is used as-is.
    """
    if os.path.sep in name or os.path.isfile(name):
        return name
    ext = os.path.splitext(name)[1].lower()
    subdir = "models" if ext == ".pt" else "converted_models"
    return os.path.join(_script_dir, subdir, name)


# EVAL_MODEL=pd runs a hand-written PD controller instead of a model, on the SAME
# 9 observations, action set, delay and sensor errors. It is the environment
# check: if this balances, the robot model, IMU, filter, signs and odometry are
# sound and any failure is the learning's. Gains: EVAL_PD="kp,kd,kx,kv".
USE_PD     = _EVAL_MODEL.lower() == "pd"
PD_GAINS   = tuple(float(v) for v in os.environ.get("EVAL_PD", "80,4,15,50").split(","))
MODEL_PATH = "pd" if USE_PD else _resolve_model_path(_EVAL_MODEL)
MODEL_FMT  = "pd" if USE_PD else os.path.splitext(MODEL_PATH)[1].lower()   # ".pt" / ".onnx" / ".tflite"
if MODEL_FMT not in (".pt", ".onnx", ".tflite", "pd"):
    sys.exit(f"error: EVAL_MODEL={_EVAL_MODEL!r} has unrecognised extension "
              f"{MODEL_FMT!r} — expected .pt, .onnx or .tflite")
DEVICE     = "cuda:0"

DT            = 0.005         # control period (200 Hz) — matches train.py
PHYS_DT       = 0.005         # physics/IMU (200 Hz)
CONTROL_DECIM = round(DT / PHYS_DT)

# N20 on an 8 V pack: 120 RPM * 8/12 = 8.4 rad/s no-load, ~5.7 loaded.
# Sets the 12.7° recoverable tilt (speed-limited). See train.py.
MAX_VEL   = 5.7
ACTION_DELAY = 2       # control steps of MCU latency = 10 ms at 200 Hz
# The queue holds exactly ACTION_DELAY entries: append-then-pop over an N-entry
# queue delays by N steps.

WHEEL1_SIGN = -1.0
WHEEL2_SIGN = -1.0

# 8 physical values + the currently-applied command.
OBS_PHYSICAL = 8
OBS_DIM      = OBS_PHYSICAL + 1
WHEEL_RADIUS = 0.030   # 60 mm wheel x 10 mm wide
DRIFT_LIMIT  = 0.5     # must match train.py — also normalises obs[7]
PITCH_LIMIT  = float(os.environ.get("EVAL_PITCH_LIMIT", "0.3"))  # must match training;
# a .pt checkpoint overrides this with the value it recorded (see LOAD MODEL).
# For a .tflite/.onnx from an OLD model (trained before 0.3), set EVAL_PITCH_LIMIT=1.2.
RATE_LIMIT   = 10.0
LIN_VEL_LIM  = 0.19    # 0.030 * 5.7 = 0.171 m/s top speed + ~11% headroom
VEL_LIMIT    = MAX_VEL

# IMU mount + spawn geometry (see train.py for the full derivation).
# adapt_onshape_usd.py puts the Imu_Sensor prim at the real MPU6050 mounting
# height (85 mm), which resolves to 27.8 mm from the Base COM — a real mounting
# offset worth only 7% of gravity at a 5 rad/s fall, so no correction is applied.
IMU_OFFSET_POS  = (0.0, 0.0, 0.0)
SPAWN_Z = 0.06350      # settled resting height — no drop on reset
IMU_SEED_PHYS_STEPS = 2
START_TILT_DEG = float(os.environ.get("EVAL_START_TILT_DEG", "5.0"))  # must match train.py
# Settled filter at episode start — must match train.py (see the note there).
FILTER_START_SETTLED   = os.environ.get("EVAL_FILTER_SETTLED", "1") == "1"
FILTER_START_NOISE_DEG = 0.5

FALL_ANGLE = math.radians(15)   # matches train.py

# Episode cap — matches train.py's MAX_STEPS_EP so "TIMEOUT" here means exactly
# what it means in the training log: survived the full episode. Without this the
# eval runs episodes indefinitely, which is a HARDER test than training ever
# posed (the policy has never seen a state beyond step 500) and makes a healthy
# policy look broken. Set EVAL_ENDURANCE=1 to remove the cap and measure how
# long it can really go.
# EVAL_EPISODE_S sets the cap in seconds (default 5, matching training). Raise
# it (e.g. 60) to see whether "successful" 5 s episodes are really stable or are
# slowly creeping toward a fall the 5 s cap hides. EVAL_ENDURANCE=1 removes the
# cap entirely — avoid combining that with EVAL_EPISODES, since a policy that
# never falls would never finish an episode.
MAX_STEPS_EP = (0 if os.environ.get("EVAL_ENDURANCE", "0") == "1"
                else round(float(os.environ.get("EVAL_EPISODE_S", "5")) / DT))

# Sensor noise. train.py trains with add_noise=True on both the IMU filter and
# the wheel encoders, so matching it is the fair comparison; EVAL_NOISE=0 gives
# the cleaner "ideal sensor" run.
ADD_NOISE = os.environ.get("EVAL_NOISE", "1") == "1"
# Proof that the policy really uses the IMU (ablation). The policy's ONLY tilt
# information is the complementary filter fed by the simulated accelerometer +
# gyro; break that path and a policy that depends on it must fall.
#   EVAL_IMU=dead   the filter gets no readings: tilt and rates read 0 forever
#   EVAL_IMU=flip   the filter's pitch and pitch rate reach the policy sign-flipped
#   EVAL_IMU=live   normal (default)
# Wheel speeds, position and the applied command are untouched in all three.
IMU_MODE = os.environ.get("EVAL_IMU", "live")

# Random pushes — OFF by default.
#   EVAL_PUSH=0.12        turn pushes on; each push is a sudden change of the
#                         robot's forward speed by a random amount between
#                         -0.12 and +0.12 m/s (random size AND direction).
#   EVAL_PUSH_EVERY_S=1.5 average time between pushes; each gap is random,
#                         between half and one-and-a-half times this.
#   EVAL_PUSH=0.03,0.06,0.09,0.12,0.15
#                         SWEEP: several intensities in one run. EVAL_EPISODES
#                         episodes at each (default 20), and every push is
#                         EXACTLY that strong, in a random direction — so each
#                         row of the table printed at the end is one intensity.
#   EVAL_DISTURB=1        older switch, same as EVAL_PUSH=0.12.
# Scale: top wheel speed is 0.171 m/s, and a push of v is worth about
# v / 0.77 rad of lean — 0.12 m/s = 8.9 deg of the 12.7 deg the robot can
# recover. Training pushes go up to 0.12; anything above ~0.17 cannot be
# recovered by any policy.
PUSH_LEVELS  = [float(v) for v in os.environ.get(
    "EVAL_PUSH", "0.12" if os.environ.get("EVAL_DISTURB", "0") == "1" else "0").split(",")]
PUSH_SWEEP   = len(PUSH_LEVELS) > 1
PUSH_MAX     = PUSH_LEVELS[0]        # current intensity; advances during a sweep
PUSH_EVERY_S = float(os.environ.get("EVAL_PUSH_EVERY_S", "1.5"))
DISTURB      = max(PUSH_LEVELS) > 0.0


def _next_push_gap() -> int:
    """Control steps until the next push: random, 0.5x to 1.5x PUSH_EVERY_S."""
    return max(1, round(random.uniform(0.5, 1.5) * PUSH_EVERY_S / DT))


# Encoder velocity quantisation — MUST match train.py and the firmware.
# 4250 counts/rev measured on the built robot; resolution is set by the sampling
# WINDOW, not the control rate (5 ms -> 0.296 rad/s, 20 ms -> 0.074 rad/s).
ENCODER_CPR      = 4250.0
ENCODER_WINDOW_S = 0.020
VEL_QUANTUM      = 1.0 / ((ENCODER_CPR / (2.0 * math.pi)) * ENCODER_WINDOW_S)

# Odometry — MUST match train.py. train.py dead-reckons x_disp by integrating
# vel_est and decaying it (integrate THEN decay), exactly as the firmware does.
# This script used to feed ground-truth position instead, so every model trained
# after that change was evaluated on an observation it never saw in training.
XDISP_TAU_S         = 20.0
XDISP_DECAY         = math.exp(-DT / XDISP_TAU_S)
RANDOMIZE_ODOM_BIAS = 0.004     # m/s; train.py draws one per env, here one per episode
# Per-episode tilt-sensor offsets — must match train.py (see the note there).
# Applied only with EVAL_NOISE=1, like the other sensor imperfections.
SENSOR_ROLL_OFFSET_DEG  = 5.0
SENSOR_PITCH_OFFSET_DEG = 1.0
_pitch_offset = 0.0
_roll_offset  = 0.0
_x_disp    = 0.0
_odom_bias = 0.0

LOG_EVERY = 100
# EVAL_VERBOSE=1 also prints a row of tilt / wheel speed / command every
# LOG_EVERY steps. Off by default: the per-push and per-episode lines below say
# more in less space.
VERBOSE = os.environ.get("EVAL_VERBOSE", "0") == "1"

# ── Observation recorder (for convert_tflite.py int8 calibration) ───────────
# int8 quantisation needs a representative sample of REAL inputs to calibrate
# activation ranges. Uniform random noise across [-1, 1] covers the whole
# observation box, but a balancing robot spends nearly all its time in a
# narrow band near 0 — calibrating on noise wastes int8's 256 levels on states
# the policy never visits and starves the ones it lives in.
# EVAL_RECORD_OBS=path/to/file.npy dumps every obs this run actually sees
# (greedy, real physics, real IMU noise) to a .npy convert_tflite.py can load
# with -q int8 --calib path/to/file.npy.
RECORD_OBS_PATH = os.environ.get("EVAL_RECORD_OBS", "")
_recorded_obs: list = []
# Recording has no natural stopping point — the main loop otherwise runs until
# simulation_app.is_running() goes false, which headless means "forever" or
# until an external kill. A hard kill (SIGKILL/timeout) never reaches the
# save-on-exit code below, so nothing gets written. Cap the recording run at a
# step count instead, and exit through the normal path once it's hit.
RECORD_STEPS = int(os.environ.get("EVAL_RECORD_STEPS", "0"))  # 0 = no cap

# Stop after this many episodes and print a FINAL summary line (0 = run until
# the window is closed). Prefer this to wrapping the script in `timeout`: a
# killed process never flushes its buffered stdout when piped, so
# `timeout ... | grep` silently prints nothing, and a time budget gives each
# model a DIFFERENT number of episodes (good models survive longer and so finish
# fewer), which makes success rates hard to compare.
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "0"))
if PUSH_SWEEP and EVAL_EPISODES == 0:
    EVAL_EPISODES = 20              # per intensity

# =====================================================================
# LOAD POLICY — .pt, .onnx, .tflite, or the PD controller
# =====================================================================
# Whatever is loaded is wrapped in an adapter exposing command(obs) -> wheel
# command in rad/s, so the main loop has one call site.
print(f"\n[LOADING POLICY] {MODEL_PATH}  ({MODEL_FMT})")


def _to_cmd(y: float) -> float:
    """Model output in [-1, 1] -> wheel command in rad/s (clamped)."""
    return max(-1.0, min(1.0, float(y))) * MAX_VEL


class _TorchPolicy:
    def __init__(self, actor):
        self._actor = actor.to(DEVICE)

    def command(self, obs: torch.Tensor) -> float:
        with torch.no_grad():
            return _to_cmd(self._actor(obs.unsqueeze(0)).item())


class _OnnxPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._in_name = self._session.get_inputs()[0].name

    def command(self, obs: torch.Tensor) -> float:
        x = obs.unsqueeze(0).cpu().numpy().astype(np.float32)
        return _to_cmd(self._session.run(None, {self._in_name: x})[0].reshape(-1)[0])


class _TFLitePolicy:
    """Handles float and fully-integer graphs: quantises / dequantises whenever
    the declared input/output dtype says to."""

    def __init__(self, path: str):
        import tensorflow as tf
        self._interp = tf.lite.Interpreter(model_path=path)
        self._interp.allocate_tensors()
        self._in_det  = self._interp.get_input_details()[0]
        self._out_det = self._interp.get_output_details()[0]

    def command(self, obs: torch.Tensor) -> float:
        x = obs.unsqueeze(0).cpu().numpy().astype(np.float32)
        idt = np.dtype(self._in_det["dtype"])
        if np.issubdtype(idt, np.integer):
            scale, zero = self._in_det["quantization"]
            x = np.clip(np.round(x / scale + zero),
                        np.iinfo(idt).min, np.iinfo(idt).max).astype(idt)
        self._interp.set_tensor(self._in_det["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out_det["index"]).reshape(-1).astype(np.float32)
        if np.issubdtype(np.dtype(self._out_det["dtype"]), np.integer):
            scale, zero = self._out_det["quantization"]
            y = (y - zero) * scale
        return _to_cmd(y[0])


class _PDPolicy:
    """Hand-written PD on the policy's own observation."""

    def command(self, obs: torch.Tensor) -> float:
        o = obs.cpu().numpy()
        kp, kd, kx, kv = PD_GAINS
        u = -(kp * o[0] * PITCH_LIMIT + kd * o[1] * RATE_LIMIT
              + kx * o[7] * DRIFT_LIMIT + kv * o[6] * LIN_VEL_LIM)
        return max(-MAX_VEL, min(MAX_VEL, float(u)))


def _check_shape(n_in: int, n_out: int) -> None:
    if (n_in, n_out) != (OBS_DIM, 1):
        print(f"\n  MODEL MISMATCH — this file is {n_in} inputs / {n_out} outputs; "
              f"this script runs PPO actors with {OBS_DIM} inputs / 1 output.")
        print("  (13-output files are from the old DQN version — see git history.)")
        simulation_app.close(); sys.exit(1)


if USE_PD:
    policy = _PDPolicy()
    print(f"  controller : hand-written PD  kp,kd,kx,kv = {PD_GAINS}")
elif MODEL_FMT == ".pt":
    try:
        _actor, ckpt = load_actor(MODEL_PATH)
    except ValueError as e:
        print(f"\n  {e}"); simulation_app.close(); sys.exit(1)
    _check_shape(_actor.net[0].in_features, 1)
    # Use the normalisation the checkpoint was trained with.
    PITCH_LIMIT = float(ckpt.get("obs_limits", {}).get("PITCH_LIMIT", PITCH_LIMIT))
    policy = _TorchPolicy(_actor)
    print(f"  iteration : {ckpt.get('iteration', '?')}")
elif MODEL_FMT == ".onnx":
    policy = _OnnxPolicy(MODEL_PATH)
    _check_shape(policy._session.get_inputs()[0].shape[-1],
                 policy._session.get_outputs()[0].shape[-1])
    print("  runtime : onnxruntime (CPU)")
else:
    policy = _TFLitePolicy(MODEL_PATH)
    _check_shape(policy._in_det["shape"][-1], policy._out_det["shape"][-1])
    print(f"  runtime : tf.lite.Interpreter  input dtype={policy._in_det['dtype'].__name__}")

# =====================================================================
# SIM + ROBOT — same construction as train.py / test_imu_tracking.py
# =====================================================================
sim = SimulationContext(SimulationCfg(dt=PHYS_DT, device=DEVICE))
sim.set_camera_view(eye=[1.2, -1.2, 0.8], target=[0.0, 0.0, 0.2])

ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=0.8, restitution=0.0))
ground_cfg.func("/World/GroundPlane", ground_cfg)

robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, disable_gravity=False),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force", ensure_drives_exist=True),
    ),
    actuators={
        "wheel_drives": ImplicitActuatorCfg(
            joint_names_expr=["Motor1Wheel1", "Motor2Wheel2"],
            stiffness=0.0, damping=500.0, effort_limit_sim=5.0),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SPAWN_Z),
        rot=(0.0, 0.0, 0.0, 1.0),          # xyzw identity (IL 3.0)
        joint_pos={"Motor1Wheel1": 0.0, "Motor2Wheel2": 0.0},
        joint_vel={"Motor1Wheel1": 0.0, "Motor2Wheel2": 0.0}),
)
robot = Articulation(robot_cfg)

# Sensor before sim.reset(); mounted at the COM to match training.
imu = Imu(ImuCfg(
    prim_path="/World/Robot/SelfBalanceRobot/Base/Imu_Sensor",
    offset=ImuCfg.OffsetCfg(pos=IMU_OFFSET_POS)))

sim.reset()
if not imu.is_initialized:
    print("✗ IMU failed to initialize"); simulation_app.close(); sys.exit(1)

# ── Robot collision material — matches train.py ──────────────────────────────
# The USD authors static=0.0 / dynamic=0.0 on every collision shape; train.py
# overwrites them to 1.0/0.8 in its domain-randomization block (which runs even
# when DOMAIN_RANDOMIZATION=False, since the nominal values are still written).
# Set here so eval and training see identical contact parameters.
#
# Measured impact: NONE on success rate (A/B tested at 100% either way). PhysX
# combines the two contacting materials, so the ground plane's 1.0/0.8 supplies
# grip even when the robot's own material is 0. Kept for exact fidelity with
# training, not because it fixes anything — do not cite it as a bug.
# (EVAL_SKIP_FRICTION=1 reproduces the old broken behaviour, for A/B testing.)
if os.environ.get("EVAL_SKIP_FRICTION", "0") != "1":
    _zero_env_id = wp.array([0], dtype=wp.int32, device="cpu")
    _materials = wp.to_torch(robot.root_physx_view.get_material_properties())
    _materials[..., 0] = 1.0    # BASE_STATIC_FRICTION
    _materials[..., 1] = 0.8    # BASE_DYNAMIC_FRICTION
    _materials[..., 2] = 0.0    # restitution
    robot.root_physx_view.set_material_properties(
        wp.from_torch(_materials.contiguous(), dtype=wp.float32), _zero_env_id)
    print("  robot friction set to static=1.0 dynamic=0.8 (matches train.py)")
else:
    print("  !! friction NOT set — using the USD's authored 0.0/0.0 (broken mode)")

BASE_IDX   = robot.body_names.index("Base")
wheel1_idx = robot.joint_names.index("Motor1Wheel1")
wheel2_idx = robot.joint_names.index("Motor2Wheel2")
filt = ComplementaryFilter(dt=PHYS_DT)

print("\n" + "=" * 62)
print("  TRAINED POLICY — closed-loop evaluation")
print(f"  model={os.path.basename(MODEL_PATH)}  action=continuous ±{MAX_VEL} rad/s  "
      f"fall>{math.degrees(FALL_ANGLE):.0f}°  "
      + (f"push sweep {PUSH_LEVELS} m/s, {EVAL_EPISODES} episodes each, about every {PUSH_EVERY_S} s"
         if PUSH_SWEEP else
         f"pushes=±{PUSH_MAX} m/s about every {PUSH_EVERY_S} s" if DISTURB else "pushes=off"))
print(f"  PITCH_LIMIT={PITCH_LIMIT}  start_tilt=±{START_TILT_DEG}°  filter_start_settled={FILTER_START_SETTLED}  "
      f"episode_cap={MAX_STEPS_EP * DT:.0f}s")
if IMU_MODE != "live":
    print(f"  *** IMU ABLATION: EVAL_IMU={IMU_MODE} — the policy is expected to FALL ***")
print(f"  gyro_bias=±{GYRO_BIAS_MAX if ADD_NOISE else 0} rad/s  pitch_off=±{SENSOR_PITCH_OFFSET_DEG if ADD_NOISE else 0}°  "
      f"roll_off=±{SENSOR_ROLL_OFFSET_DEG if ADD_NOISE else 0}°")
print("=" * 62 + "\n")


# =====================================================================
# HELPERS — get_obs is byte-for-byte train.py's, with add_noise=False
# =====================================================================

def get_obs(applied_cmd: float) -> torch.Tensor:
    jvel = robot.data.joint_vel[0]
    pos  = robot.data.body_pos_w[0, BASE_IDX]
    pitch, roll = filt.pitch, filt.roll
    prate, rrate = filt.pitch_rate, filt.roll_rate
    if IMU_MODE == "dead":
        pitch = roll = prate = rrate = 0.0
    elif IMU_MODE == "flip":
        pitch, prate = -pitch, -prate

    # Wheel encoder noise — train.py applies VEL_NOISE=0.02 here when add_noise.
    if ADD_NOISE:
        w1 = jvel[wheel1_idx].item() * WHEEL1_SIGN + random.gauss(0, 0.02)
        w2 = jvel[wheel2_idx].item() * WHEEL2_SIGN + random.gauss(0, 0.02)
    else:
        w1 = jvel[wheel1_idx].item() * WHEEL1_SIGN
        w2 = jvel[wheel2_idx].item() * WHEEL2_SIGN
    # Encoder staircase — see train.py. Applied even when ADD_NOISE is off,
    # because quantisation is hardware, not noise. EVAL_NOISE=0 gives an ideal
    # ANALOG sensor, never an infinitely fine encoder.
    if VEL_QUANTUM > 0.0:
        w1 = round(w1 / VEL_QUANTUM) * VEL_QUANTUM
        w2 = round(w2 / VEL_QUANTUM) * VEL_QUANTUM

    # Dead-reckoned x_disp, byte-for-byte train.py's logic: the bias enters
    # vel_est itself (as gyro bias does on hardware, through prate), so it shows
    # up in obs[6] AND integrates into obs[7].
    global _x_disp
    vel_est = (-0.5 * (w1 + w2) + prate) * WHEEL_RADIUS
    if ADD_NOISE:
        vel_est += _odom_bias
    _x_disp = (_x_disp + vel_est * DT) * XDISP_DECAY

    def _c(v, lim):
        return max(-1.0, min(1.0, float(v) / lim))

    return torch.tensor([
        _c(pitch + _pitch_offset, PITCH_LIMIT),
        _c(prate, RATE_LIMIT),
        _c(roll + _roll_offset,  PITCH_LIMIT),
        _c(rrate, RATE_LIMIT),
        _c(w1, VEL_LIMIT),
        _c(w2, VEL_LIMIT),
        _c(vel_est, LIN_VEL_LIM),
        _c(_x_disp, DRIFT_LIMIT),
        _c(applied_cmd, MAX_VEL),
    ], device=DEVICE, dtype=torch.float32)


def true_pitch_roll():
    q = robot.data.body_quat_w[0, BASE_IDX]      # xyzw
    qx, qy, qz, qw = (q[i].item() for i in range(4))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    roll  = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    return pitch, roll


def apply_action(vel: float):
    robot.set_joint_velocity_target(
        torch.tensor([[WHEEL1_SIGN * vel, WHEEL2_SIGN * vel]],
                     device=DEVICE, dtype=torch.float32))


def is_done():
    pitch, roll = true_pitch_roll()
    pos = robot.data.body_pos_w[0, BASE_IDX]
    return (abs(pitch) > FALL_ANGLE or abs(roll) > FALL_ANGLE
            or abs(pos[0].item()) > DRIFT_LIMIT)


def stage_reset():
    """Teleport to spawn with a small pitch perturbation, resetting the filter."""
    perturb = random.uniform(-math.radians(START_TILT_DEG), math.radians(START_TILT_DEG))
    root = torch.zeros((1, 7), device=DEVICE)
    root[0, 2] = SPAWN_Z
    root[0, 4] = math.sin(perturb / 2)   # qy
    root[0, 6] = math.cos(perturb / 2)   # qw
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=DEVICE))
    n = robot.num_joints
    robot.write_joint_position_to_sim_index(position=torch.zeros((1, n), device=DEVICE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros((1, n), device=DEVICE))
    filt.reset()
    # Per-episode constant gyro bias, as train.py. Before set_attitude.
    filt.set_gyro_bias(*((random.uniform(-GYRO_BIAS_MAX, GYRO_BIAS_MAX),
                          random.uniform(-GYRO_BIAS_MAX, GYRO_BIAS_MAX))
                         if ADD_NOISE else (0.0, 0.0)))
    if FILTER_START_SETTLED:
        # Bias offset 63..100% built at arm, as train.py.
        filt.set_attitude(perturb + random.gauss(0.0, math.radians(FILTER_START_NOISE_DEG)),
                          settled=random.uniform(filt.settle_fraction(ARM_DELAY_S), 1.0))
    # The odometry integral restarts with the episode, as in train.py and as the
    # firmware zeroes it on arm. New bias per episode for coverage.
    global _x_disp, _odom_bias, _pitch_offset, _roll_offset
    _pitch_offset = (math.radians(random.uniform(-SENSOR_PITCH_OFFSET_DEG, SENSOR_PITCH_OFFSET_DEG))
                     if ADD_NOISE else 0.0)
    _roll_offset = (math.radians(random.uniform(-SENSOR_ROLL_OFFSET_DEG, SENSOR_ROLL_OFFSET_DEG))
                    if ADD_NOISE else 0.0)
    _x_disp = 0.0
    _odom_bias = (random.uniform(-RANDOMIZE_ODOM_BIAS, RANDOMIZE_ODOM_BIAS)
                  if RANDOMIZE_ODOM_BIAS > 0 else 0.0)
    # Flush the pre-fall wheel command. Zeroing joint VELOCITY does not clear the
    # drive TARGET, so without this the actuator keeps driving at whatever it was
    # commanding when the robot fell (often +/-4 rad/s) straight through the seed
    # substeps below — kicking the robot over the moment it respawns. train.py
    # gets this for free by flushing env_action_buf to zeros, so its
    # next iteration commands zero before stepping.
    apply_action(0.0)


def step_physics(seed_filter: bool):
    """Advance one control step of physics.

    On a reset, run at least IMU_SEED_PHYS_STEPS and seed the filter on the last
    of them: the IMU's first sample after a teleport differences against the
    PRE-teleport velocity and spikes, which would make the accel-only seed
    collapse to "upright". At 200 Hz control CONTROL_DECIM is 1, so the extra
    step has to be taken explicitly rather than found inside the substep loop.
    """
    n = max(CONTROL_DECIM, IMU_SEED_PHYS_STEPS) if seed_filter else CONTROL_DECIM
    for sub in range(n):
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYS_DT)
        imu.update(PHYS_DT, force_recompute=True)
        filt.update_from_imu(imu, add_noise=ADD_NOISE,
                             init=(seed_filter and not FILTER_START_SETTLED
                                   and sub == IMU_SEED_PHYS_STEPS - 1))


def apply_disturbance():
    # Sweep: exactly the current intensity, random direction. Otherwise a random
    # size up to PUSH_MAX.
    kick = (random.choice([-1.0, 1.0]) * PUSH_MAX if PUSH_SWEEP
            else random.uniform(-PUSH_MAX, PUSH_MAX))
    vel = robot.data.root_vel_w.clone()
    vel[0, 0] += kick
    robot.write_root_velocity_to_sim(vel)
    return kick


# =====================================================================
# MAIN LOOP
# =====================================================================
stage_reset()
step_physics(seed_filter=True)

action_buf = [0.0] * ACTION_DELAY
applied = 0.0
obs = get_obs(applied)

ep, ep_step = 0, 0
timeouts = 0
survivals = []
next_disturb = _next_push_gap()
step = 0
ep_peak = 0.0                 # largest |tilt| this episode, degrees
ep_pushes = ep_recovered = 0  # pushes given / recovered from, this episode
push = None                   # the push still being judged: dict(kick, t, peak)
# Per-intensity totals: episodes, episodes survived, survival time, pushes, pushes recovered
lvl = dict(ep=0, ok=0, surv=0.0, pushes=0, rec=0)
level_idx = 0
level_rows = []
MODEL_NAME = os.path.basename(MODEL_PATH)


def say(msg: str) -> None:
    print(msg, flush=True)


def lean_deg(v: float) -> float:
    """Sudden tilt a speed kick of v m/s is worth (v / sqrt(l_eff * g))."""
    return math.degrees(abs(v) / 0.770)


def close_push(outcome: str) -> None:
    """Print the verdict on the push being judged."""
    global push, ep_recovered
    if push is None:
        return
    if outcome == "recovered":
        ep_recovered += 1
        lvl["rec"] += 1
    say(f"  PUSH  ep {ep + 1:<3d} t={push['t']:4.2f}s  {push['kick']:+.3f} m/s "
        f"(~{lean_deg(push['kick']):4.1f}° of lean)  ->  {outcome:<22s} "
        f"peak tilt {push['peak']:4.1f}°")
    push = None


def level_line(tag: str) -> str:
    n = max(lvl["ep"], 1)
    pr = f"{lvl['rec']}/{lvl['pushes']}" if lvl["pushes"] else "-"
    return (f"{tag} ±{PUSH_MAX:.3f} m/s : episodes survived {lvl['ok']}/{lvl['ep']} = "
            f"{100.0 * lvl['ok'] / n:3.0f}%   pushes recovered {pr}   "
            f"mean survival {lvl['surv'] / n:.2f}s")


if DISTURB:
    say("\nPUSH GUIDE (this robot): top wheel speed 0.171 m/s; it can recover about 12.7° of sudden lean")
    say("   0.03 gentle (2°) | 0.06 light (4°) | 0.09 firm (7°) | 0.12 training max (9°) | "
        "0.15 beyond training (11°)")
    say("   0.17 theoretical limit (13°) | 0.20-0.30 extreme: expected to fall, shows HOW it fails")
    if PUSH_SWEEP:
        say(f"\n=== INTENSITY 1/{len(PUSH_LEVELS)} : ±{PUSH_MAX:.3f} m/s "
            f"(~{lean_deg(PUSH_MAX):.1f}° of lean), {EVAL_EPISODES} episodes ===")
if VERBOSE:
    say(f"{'Step':>6} {'true_pitch':>10} {'est_pitch':>10} {'roll':>7} "
        f"{'wheel':>7} {'action':>7} {'epStep':>7}")

while simulation_app.is_running():
    # The policy's command (no exploration noise), then apply the delayed one
    # (matches training's ACTION_DELAY).
    action_buf.append(policy.command(obs))
    applied = action_buf.pop(0)
    apply_action(applied)

    if PUSH_MAX > 0.0 and step >= next_disturb:
        close_push("recovered")            # it was still up when the next one came
        push = dict(kick=apply_disturbance(), t=ep_step * DT, peak=0.0)
        ep_pushes += 1
        lvl["pushes"] += 1
        next_disturb = step + _next_push_gap()

    step_physics(seed_filter=False)
    obs = get_obs(applied)

    if RECORD_OBS_PATH:
        _recorded_obs.append(obs.cpu().numpy().copy())
        if RECORD_STEPS > 0 and len(_recorded_obs) >= RECORD_STEPS:
            break

    tp, tr = true_pitch_roll()
    tilt = abs(math.degrees(tp))
    ep_peak = max(ep_peak, tilt)
    if push is not None:
        push["peak"] = max(push["peak"], tilt)
    ep_step += 1

    if VERBOSE and step % LOG_EVERY == 0:
        jv = robot.data.joint_vel[0]
        wheel = (jv[wheel1_idx].item() * WHEEL1_SIGN + jv[wheel2_idx].item() * WHEEL2_SIGN) / 2
        say(f"{step:6d} {math.degrees(tp):+10.2f} {math.degrees(filt.pitch):+10.2f} "
            f"{math.degrees(tr):+7.2f} {wheel:+7.2f} {applied:+7.2f} {ep_step:7d}")

    fell = is_done()
    timed_out = MAX_STEPS_EP > 0 and ep_step >= MAX_STEPS_EP
    if fell or timed_out:
        # A "fall" with the robot still upright is the drift limit (0.5 m from start).
        how = ("HELD" if not fell else
               "FELL" if tilt >= math.degrees(FALL_ANGLE) - 1.0 else "DRIFTED 0.5 m")
        if push is not None:
            close_push("recovered" if not fell else
                       f"{how} {ep_step * DT - push['t']:.2f}s later")
        ep += 1
        survivals.append(ep_step * DT)
        timeouts += 0 if fell else 1
        lvl["ep"] += 1; lvl["ok"] += 0 if fell else 1; lvl["surv"] += ep_step * DT
        pushes_txt = f"pushes recovered {ep_recovered}/{ep_pushes}   " if DISTURB else ""
        say(f"EP {ep:3d}  {how:<13s} {ep_step * DT:5.2f}s   {pushes_txt}peak tilt {ep_peak:4.1f}°   "
            f"| so far {timeouts}/{ep} = {100.0 * timeouts / ep:.0f}%")
        stage_reset()
        step_physics(seed_filter=True)
        action_buf = [0.0] * ACTION_DELAY
        applied = 0.0
        obs = get_obs(applied)
        ep_step = 0
        ep_peak = 0.0
        ep_pushes = ep_recovered = 0
        next_disturb = step + _next_push_gap()
        if PUSH_SWEEP:
            if lvl["ep"] >= EVAL_EPISODES:
                say(level_line("LEVEL DONE"))
                level_rows.append((PUSH_MAX, dict(lvl)))
                level_idx += 1
                lvl = dict(ep=0, ok=0, surv=0.0, pushes=0, rec=0)
                if level_idx >= len(PUSH_LEVELS):
                    break
                PUSH_MAX = PUSH_LEVELS[level_idx]
                say(f"\n=== INTENSITY {level_idx + 1}/{len(PUSH_LEVELS)} : ±{PUSH_MAX:.3f} m/s "
                    f"(~{lean_deg(PUSH_MAX):.1f}° of lean), {EVAL_EPISODES} episodes ===")
        elif EVAL_EPISODES and ep >= EVAL_EPISODES:
            break

    step += 1

if level_rows:
    say(f"\nPUSH SWEEP  {MODEL_NAME}  (every push exactly this strong, random direction, "
        f"about every {PUSH_EVERY_S} s)")
    say(f"  {'push m/s':>9} {'~lean':>7} {'episodes survived':>19} {'pushes recovered':>18} "
        f"{'mean survival':>14}")
    for lv, d in level_rows:
        pr = (f"{d['rec']}/{d['pushes']} {100.0 * d['rec'] / d['pushes']:3.0f}%"
              if d["pushes"] else "-")
        say(f"  {lv:>9.3f} {lean_deg(lv):>6.1f}° {d['ok']:>10d}/{d['ep']:<3d} "
            f"{100.0 * d['ok'] / d['ep']:>3.0f}% {pr:>18s} {d['surv'] / d['ep']:>13.2f}s")
elif DISTURB and lvl["ep"]:
    say("\n" + level_line("PUSHES"))

if survivals:
    # One line, flushed, so `| grep FINAL` works even through a pipe and even if
    # simulation_app.close() exits the process hard afterwards.
    print(f"\nFINAL  {os.path.basename(MODEL_PATH)}  SUCCESS RATE {timeouts}/{ep} = "
          f"{100.0 * timeouts / ep:.0f}%   mean survival "
          f"{sum(survivals) / len(survivals):.2f}s", flush=True)

if RECORD_OBS_PATH and _recorded_obs:
    arr = np.stack(_recorded_obs).astype(np.float32)
    np.save(RECORD_OBS_PATH, arr)
    print(f"\n[RECORDED] {arr.shape[0]} real observations -> {RECORD_OBS_PATH}")
    print(f"           pass to convert_tflite.py with -q int8 --calib {RECORD_OBS_PATH}")

simulation_app.close()
