// robot_config.h — every constant the deployed policy depends on, in one place.
//
// The values under "POLICY CONTRACT" are not tunable. They are what train.py
// used to build the observation vector, and the network learned against exactly
// these. Change one and the model is being fed something it has never seen.
//
// The values under "HARDWARE" are yours to measure and tune — they describe the
// physical robot, not the policy.
#pragma once

#include <Arduino.h>

// =====================================================================
// DEBUG PRINTS
// =====================================================================
// Every Serial.print in the codebase that is NOT a safety event (fallen,
// watchdog, a failed Invoke()) goes through DBG_PRINTF / DBG_PRINTLN instead of
// calling Serial directly. Serial.printf on this board can take from tens of us
// to low milliseconds depending on the USB buffer, and MODE_RUN's control loop
// has a 5 ms total budget — an unconditional print in that path can BE the
// overrun it would otherwise report.
//
// DEBUG_PRINTS defaults from ROBOT_MODE: on for MODE_DIAG/MODE_VALIDATE, where
// nothing is time-critical and the prints are the point, off for MODE_RUN,
// where they cost real budget. Override either way with a build flag:
//   -DDEBUG_PRINTS=1   force prints on even in MODE_RUN (bench debugging only)
//   -DDEBUG_PRINTS=0   force prints off even in MODE_DIAG/VALIDATE
//
// ROBOT_MODE itself is defined by the build environment in platformio.ini
// (policy_quick_check=0 DIAG, policy_model_validate=1 VALIDATE, policy=2 RUN),
// so this must come after that -D but doesn't need main.cpp's MODE_* constants —
// the raw values line up (DIAG=0, VALIDATE=1, RUN=2).
#ifndef DEBUG_PRINTS
  #if defined(ROBOT_MODE) && ROBOT_MODE == 2   // MODE_RUN
    #define DEBUG_PRINTS 0
  #else
    #define DEBUG_PRINTS 1
  #endif
#endif

#if DEBUG_PRINTS
  #define DBG_PRINTLN(...)  Serial.println(__VA_ARGS__)
  #define DBG_PRINTF(...)   Serial.printf(__VA_ARGS__)
#else
  #define DBG_PRINTLN(...)  do {} while (0)
  #define DBG_PRINTF(...)   do {} while (0)
#endif

// =====================================================================
// POLICY CONTRACT — mirrors train.py / imu_filter.py. Do not tune.
// =====================================================================

#define OBS_DIM_CFG    10
#define N_OUTPUTS_CFG  1

// Contract of the linked model: PPO policy model_h64_it0250 (2026-10-11 run:
// 10 inputs, encoder-count x_disp, position integral, DRIFT_LIMIT 0.3),
// training repo README.md section 7. Every value in this section must match
// the checkpoint behind src/policy/model_data.cpp, and they change only in the
// same step as a new model. Models before this one (DQN, 13 discrete actions,
// PITCH_LIMIT 1.2, DRIFT_LIMIT 2.0, no x_disp leak) do not run on this
// firmware; the pre-PPO state is in git stash "pre-PPO firmware".

// The policy acts every 5 ms and its notion of "how fast things change" is
// baked in at that rate.
constexpr float  CONTROL_DT_S  = 0.005f;
constexpr uint32_t CONTROL_DT_US = 5000;

// The network outputs ONE number in [-1, 1] (hard-clipped inside the graph).
// Wheel command = output * MAX_VEL, the same for both wheels, any value in the
// range rather than one of a fixed set. 5.7 rad/s x 0.030 m = 0.171 m/s, about
// 78% of the motors' measured no-load top speed on the 8 V pack.
constexpr float MAX_VEL = 5.7f;

// The command applied now is the one computed ACTION_DELAY control steps
// earlier (10 ms), modelling MCU latency. The policy learned to compensate for
// that lag, so the firmware reproduces it; removing a delay the model expects
// makes it overcorrect.
constexpr int ACTION_DELAY = 2;

// Observation normalisation. Every obs element is value/limit clamped to
// [-1, 1].
//
// PITCH_LIMIT normalises BOTH pitch (obs[0]) and roll (obs[2]). 0.3 rad, not
// the 1.2 of pre-PPO models: the model's input layer quantises all 9 inputs
// with ONE scale (0.007843/step), so 0.3 gives 0.135 deg of tilt per step where
// 1.2 gave 0.54 deg, coarser than the band balancing happens in.
constexpr float PITCH_LIMIT  = 0.3f;    // rad
constexpr float RATE_LIMIT   = 10.0f;   // rad/s
constexpr float VEL_LIMIT    = MAX_VEL; // rad/s
constexpr float LIN_VEL_LIM  = 0.19f;   // m/s (0.030 x 5.7 = 0.171 top)

// Normalises obs[7], the policy's sense of how far it is from where it armed,
// and is also the distance at which the robot switches off (DRIFT_HOLD_MS).
// 0.3 m since model_h64_it0250 (0.5 for model_h64_it0300, 2.0 pre-PPO). In
// training an episode fails at this distance, so the policy has no trained
// behaviour beyond it.
constexpr float DRIFT_LIMIT  = 0.3f;    // m

// obs[7] x_disp comes from the encoder COUNTS since arming, not from
// integrating vel_est (model_h64_it0300 used a leaky integral, tau 20 s):
//   wheel_angle = mean signed counts since arming * 2*pi / ENCODER_COUNTS_PER_REV
//   x_disp      = WHEEL_RADIUS * (-wheel_angle + (pitch - pitch at arming))
// Nothing accumulates, so there is no leak. Policy::buildObs() does it.
//
// obs[9] x_int, the position integral: every tick
//   x_int = clamp(x_int + x_disp * dt, -XINT_LIMIT, +XINT_LIMIT),  obs[9] = x_int / XINT_LIMIT
// It is the policy's only memory, and what lets it return to the arming point
// despite a constant tilt-reading error. Zeroed at arming. The clamp is the
// anti-windup.
constexpr float XINT_LIMIT   = 0.5f;    // m*s

// Switch off when |x_disp| has stayed at or beyond DRIFT_LIMIT this long: the
// robot was lifted (wheels spinning in the air) or shoved out of range. The
// normal auto-arm then restarts it with a fresh start point.
constexpr uint32_t DRIFT_HOLD_MS = 500;

// Copy the model into internal RAM at boot instead of reading it from flash on
// every inference. Numerically identical — same bytes, different address.
// The weights would otherwise be read through the ESP32's 32 KB flash cache,
// which is shared with the TFLM kernel code also executing from flash. Costs
// g_model_len bytes of heap (~8 KB of ~330 KB free). Set false to compare.
constexpr bool MODEL_IN_RAM = true;
constexpr float WHEEL_RADIUS = 0.030f;  // m

// Encoder velocity sampling (train.py:208-210). These are CONTRACT, not tuning:
// train.py:207 says "Set ENCODER_WINDOW_S to match the firmware", and the
// simulator quantises w1/w2 to VEL_QUANTUM before normalising them.
//
// Resolution depends on the SAMPLING WINDOW, not the control rate:
//     5 ms (1 tick)  -> 0.296 rad/s   6.6 output steps of the policy
//    20 ms (4 ticks) -> 0.074 rad/s   what train.py uses
// At a 5 ms window the +/-0.114 and +/-0.285 rad/s station-keeping actions are
// below one count — exactly the commands the robot needs to hold itself still.
// The 20 ms window costs 10 ms of lag, a fifth of the 50 ms fall time constant.
constexpr float ENCODER_WINDOW_S = 0.020f;
constexpr int   ENC_WINDOW_TICKS = 4;      // 20 ms / 5 ms, must be >= 1

// 1 / ((ENCODER_CPR / 2pi) * ENCODER_WINDOW_S). Hard-quantise w1/w2 to multiples
// of this AFTER computing rad/s and BEFORE normalising. train.py:766-767: "a
// staircase is a property of the hardware, not a noise source you can switch
// off" — it is applied in the simulator even with noise disabled, so the network
// has only ever seen stepped wheel velocities. Do not smooth these.
constexpr float VEL_QUANTUM = 0.07391982714328925f;   // rad/s

// Complementary filter (imu_filter.py).
constexpr float FILTER_ALPHA          = 0.995f;

// The filter is seeded from the accelerometer only once accel trust reaches
// this, i.e. the robot is effectively still. Before that every tick retries,
// and the auto-arm countdown doesn't start, so the policy always begins with
// the filter settled at the true tilt — what the simulator's start assumes.
constexpr float FILTER_SEED_MIN_TRUST = 0.9f;

// Largest IMU mounting offset accepted, per axis. A board on a robot is off by
// a degree or two, not more; a bigger reading means the chassis was not held
// vertical when 'o' was pressed, and using it would make the robot balance
// leaning by that much. Bigger values are refused when measured AND ignored
// when loaded from flash.
constexpr float MOUNT_OFFSET_MAX_RAD = 0.0873f;   // 5 deg

// Largest balance trim accepted (see Imu::setBalanceTrim). A 45 g battery
// 20 mm off the axle is already ~10 deg; more than this means the measurement
// was not taken at the balance pose.
constexpr float BALANCE_TRIM_MAX_RAD = 0.2618f;   // 15 deg
constexpr float GRAVITY_MAG           = 9.81f;   // m/s^2
constexpr float ACCEL_TRUST_FULL_BAND = 1.0f;
constexpr float ACCEL_TRUST_ZERO_BAND = 4.0f;

// =====================================================================
// HARDWARE — measure and tune these for your build.
// =====================================================================

// ── MPU6050 ──────────────────────────────────────────────────────────
#define MPU_ADDR      0x68
#define I2C_SDA_PIN   21
#define I2C_SCL_PIN   22
// 400 kHz is the sensor's rated I2C speed (the part answers WHO_AM_I 0x70, an
// MPU6500). 1 MHz was tried and read without errors, saving ~200 us per tick,
// but it is out of spec and the time is no longer needed (the tick is ~2.4 ms
// of 5 ms).
//
// The gyro's zero offset depends on how the sensor is being read. Measured
// with the robot still, as the mean rate after the saved bias, in LSB (x/y/z):
//     1 MHz,   read every 5 ms after a busy-wait   +47 -49 -30
//     400 kHz, read every 5 ms after a busy-wait   +26 -26 -20
//     1 MHz,   read every ~2 ms with delay()       +10 -10  -7
//     400 kHz, read every ~2 ms with delay()       -42 +21 +18
// i.e. it moves steadily with bus/CPU activity, about 6 LSB per percent of
// bus duty. So the bias is only valid for the conditions it was measured in:
// Imu::calibrateGyro() reads at the control loop's own cadence and bus speed.
// CHANGING THIS VALUE INVALIDATES THE SAVED CALIBRATION — bump
// GYRO_CAL_VERSION so every board recalibrates.
#define I2C_FREQ_HZ   400000

// Identifies the conditions the saved gyro bias was measured under (read
// cadence, I2C speed, calibration method). A bias saved with another version
// is discarded at boot and measured again.
constexpr uint32_t GYRO_CAL_VERSION = 2;

// A calibration window is rejected if any gyro axis moved by more than this
// peak-to-peak (the robot was being handled). Sensor noise alone is ~0.02.
constexpr float GYRO_CAL_MAX_RANGE = 0.08f;   // rad/s

// Gyro full scale. This is NOT a free choice: RATE_LIMIT is 10 rad/s = 573 deg/s,
// so the default +/-250 deg/s range would clip at 4.4 rad/s and hand the policy a
// flat-topped rate signal during exactly the fast tilts it needs to catch.
// +/-1000 deg/s covers the trained range with margin.
#define GYRO_FS_SEL       2          // 0=250, 1=500, 2=1000, 3=2000 deg/s
constexpr float GYRO_LSB_PER_DPS = 32.8f;   // for FS_SEL = 2

// Accel full scale +/-2g. The filter's trust gating tops out at 9.81+4 = 13.8 m/s^2
// (1.4 g), so +/-2g covers it and gives the best tilt resolution.
#define ACCEL_FS_SEL      0          // 0=2g, 1=4g, 2=8g, 3=16g
constexpr float ACCEL_LSB_PER_G = 16384.0f; // for FS_SEL = 0

// ── Wheel encoders (N20 magnetic quadrature) ─────────────────────────
#define ENC1_A_PIN  18   // LEFT  motor C1
#define ENC1_B_PIN  19   // LEFT  motor C2
#define ENC2_A_PIN  16   // RIGHT motor C1 (RX2)
#define ENC2_B_PIN  17   // RIGHT motor C2 (TX2)

// Counts per revolution of the WHEEL (after gearbox), 4x quadrature decoding.
// 4250 is the figure your own motor test sketch divides by. It does not
// correspond to a clean PPR x ratio x 4, so confirm it by hand before trusting
// it: a wrong CPR scales every velocity and the odometry proportionally.
// VERIFY THIS: run MODE_DIAG, turn one wheel exactly one full revolution by
// hand, and read the printed raw count. Every velocity, the linear-velocity
// estimate and the odometry scale directly off this number.
constexpr float ENCODER_COUNTS_PER_REV = 4250.0f;

// Encoder direction. Positive must mean the same physical rotation a positive
// ACTION produces — the servo compares the two directly, so ENCn_SIGN and
// MOTOR_x_SIGN must always be flipped together. Getting the PAIR wrong makes
// the velocity loop run away to full duty; getting BOTH pairs inverted (the
// polarity below) makes the robot drive the wrong way relative to its lean.
//
// Polarity, set from train.py's own balance controller (train.py:1112):
//     u = -(KP*pitch + KD*pitch_rate + KX*x_disp + KV*vel)
// A positive pitch must therefore produce a NEGATIVE command, and that command
// has to move the wheels TOWARDS the lean to catch the fall. In other words a
// positive command must drive the robot BACKWARD with respect to the direction
// the firmware calls positive pitch.
// ENC2/MOTOR_B are flipped relative to ENC1/MOTOR_A because the right motor is
// mounted mirrored: with both the same, one wheel turned the opposite way and
// the robot spun on the spot instead of driving straight.
// POLARITY CONFIRMED ON THE FLOOR, 2026-10-12. Required: a NEGATIVE command
// (negative w) rolls the robot TOWARDS the side that reads positive pitch.
// That is the simulator's geometry: ground speed = R * (-w + pitch_rate), both
// rotations taken in the same sense. With the opposite polarity (ENC1/MOTOR_A
// = -1, ENC2/MOTOR_B = +1) the first floor run fell every time: armed at
// +2.8 deg, the policy commanded full negative within 100 ms and pitch then
// rose steadily to +35 deg, and the mirror image from -0.6 deg. Wheels driving
// under the lean would have reversed the tilt within ~0.1 s; it never did.
// Never flip one pair without the other three.
constexpr float ENC1_SIGN = +1.0f;
constexpr float ENC2_SIGN = -1.0f;

// Velocity smoothing was removed deliberately — do not reintroduce it. An EMA
// was here to soften the coarse per-tick difference at low speed, but the policy
// was trained on hard-quantised wheel velocities (see VEL_QUANTUM above), and
// smoothing them hands the network a different signal SHAPE than it learned on.
// The 20 ms window plus quantisation replaces it.

// ── TB6612FNG motor driver ───────────────────────────────────────────
#define MOTOR_STBY_PIN  13
#define MOTOR_A_PWM_PIN 25   // LEFT  PWMA
#define MOTOR_A_IN1_PIN 26   // LEFT  AIN1
#define MOTOR_A_IN2_PIN 27   // LEFT  AIN2
#define MOTOR_B_PWM_PIN 33   // RIGHT PWMB
#define MOTOR_B_IN1_PIN 32   // RIGHT BIN1
#define MOTOR_B_IN2_PIN 4    // RIGHT BIN2

// LEDC (Arduino core 2.0.17 uses the channel-based API).
#define MOTOR_A_LEDC_CH 0
#define MOTOR_B_LEDC_CH 1
constexpr uint32_t MOTOR_PWM_FREQ_HZ = 20000;  // above audible
constexpr uint8_t  MOTOR_PWM_BITS    = 10;     // 0..1023
constexpr int      MOTOR_PWM_MAX     = (1 << MOTOR_PWM_BITS) - 1;

// Motor direction. Flipped together with ENCn_SIGN above — see the polarity
// note there. If ONE wheel physically turns the wrong way (the robot spins on
// the spot instead of driving straight), flip only that wheel's sign here and
// its matching ENCn_SIGN.
constexpr float MOTOR_A_SIGN = +1.0f;
constexpr float MOTOR_B_SIGN = -1.0f;

// ── Per-wheel velocity controller ────────────────────────────────────
// The simulator drove the wheels with stiffness=0, damping=500 — a pure velocity
// servo with a 5 N.m torque cap. Commands are therefore velocity *targets*, not
// PWM values, and open-loop PWM will not track them under changing load. This PI
// loop plus feedforward is the hardware stand-in.
//
// TUNE THESE ON HARDWARE, wheels off the ground, using MODE_DIAG: command a
// constant velocity and compare measured against target.
// Starting points, scaled to MAX_VEL = 5.7 rad/s and a 10-bit (0..1023) duty.
// VEL_KFF is the important one: at full command the feedforward alone should be
// most of the way to full duty, so (1023 - deadband) / 5.7 ~= 170. The PI terms
// only have to clean up what the feedforward misses.
constexpr float VEL_KP  = 80.0f;    // PWM counts per (rad/s) of error
constexpr float VEL_KI  = 300.0f;   // PWM counts per (rad/s * s)
// Measured on this robot from velocity steps to 2.85 rad/s ('p' in
// policy_quick_check, 2026-10-12): standing on the floor the wheels settled
// 14-22% above the target with the old value of 141, and the duty they
// actually needed was (duty - deadband) / speed = 99 to 117 counts per rad/s
// across both wheels and two runs. Free-spinning gave 105, so load barely
// changes it. 141 came from the motortest sweep, which under-reads speed
// because it averages over spin-up and coasting.
constexpr float VEL_KFF = 110.0f;   // PWM counts per (rad/s) of target

// Anti-windup, expressed as the largest CONTRIBUTION the integral term may make,
// in PWM counts. Clamping the raw integral instead would be meaningless — its
// useful range depends entirely on VEL_KI.
constexpr float VEL_I_CLAMP_PWM = 600.0f;

// Duty below which the motor does not turn at all. Measured, not guessed: in
// MODE_DIAG ramp the duty until the wheel starts moving.
constexpr int MOTOR_DEADBAND = 37;   // from the same duty-vs-speed fit

// ── Safety ───────────────────────────────────────────────────────────
// Past this tilt the robot has fallen and the policy cannot recover it; cutting
// the motors stops it driving itself across the floor on its side.
constexpr float ABORT_PITCH_RAD = 0.7f;   // ~40 deg

// The real firmware starts itself: this long after power-on, after a 'c'
// recalibration, or after being stood back up following a fall (tilt back
// inside ABORT_PITCH_RAD), the motors arm with no key press.
//
// CONTRACT (train side calls it ARM_DELAY_S, imu_filter.py): training starts
// each episode with the filter 63-100% settled toward its gyro-bias offset,
// where 63% = 1 - exp(-AUTO_ARM_DELAY / filter tau ~1 s) is this first-arm
// delay. Changing this value changes that lower bound — tell the training side.
constexpr uint32_t AUTO_ARM_DELAY_MS = 1000;

// Arm only inside this tilt window (filtered pitch). Training starts every
// episode within +/-START_TILT_DEG = 5 deg (train.py, from 2026-10-07; +/-3 deg
// before that), so arming anywhere up to ABORT_PITCH_RAD (~40 deg) could start
// the policy in a state it has never seen, possibly past the ~20 deg it can
// recover from at all. Outside the window the motors stay off and the 1 s
// countdown restarts once it is back inside. ABORT_PITCH_RAD is a different job:
// it is the fall check AFTER arming.
constexpr float ARM_PITCH_RAD = 0.0873f;   // 5 deg

// Roll safety net, not a training-matched window: training never randomises
// roll, so this only catches a robot set down tipped onto one wheel or on a
// sideways slope. Deliberately loose — this robot reads +1.9 to +4.2 deg of
// roll sitting still on the bench (mount offset or surface), so a 5 deg gate
// would intermittently refuse to start.
constexpr float ARM_ROLL_RAD = 0.2618f;    // 15 deg

// Motors are cut if a control tick takes longer than this — a stalled inference
// or a blocked I2C read must not leave the last command running.
constexpr uint32_t WATCHDOG_US = 50000;   // 50 ms
