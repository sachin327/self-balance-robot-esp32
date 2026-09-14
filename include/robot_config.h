// robot_config.h — every constant the deployed policy depends on, in one place.
//
// The values under "POLICY CONTRACT" are not tunable. They are what train.py
// used to build the observation vector, and the network learned against exactly
// these. Change one and the model is being fed something it has never seen.
//
// The values under "HARDWARE" are yours to measure and tune — they describe the
// physical robot, not the policy.
#pragma once

// =====================================================================
// POLICY CONTRACT — mirrors train.py / imu_filter.py. Do not tune.
// =====================================================================

#define OBS_DIM_CFG    9
#define N_ACTIONS_CFG  13

// train_obs9_baseline.py DT = 0.005. The policy was trained to act every 5 ms and its notion
// of "how fast things change" is baked in at that rate.
constexpr float  CONTROL_DT_S  = 0.005f;
constexpr uint32_t CONTROL_DT_US = 5000;

// Wheel command levels, rad/s (train_obs9_baseline.py ACTIONS, MAX_VEL = 5.7).
// 5.7 rad/s x 0.030 m = 0.171 m/s, which is what the 8 V pack can actually
// deliver — the earlier 10.5 asked for 100 RPM the motors could not reach.
constexpr float MAX_VEL = 5.7f;
constexpr float ACTIONS[N_ACTIONS_CFG] = {
    -MAX_VEL, -MAX_VEL * 0.5f, -MAX_VEL * 0.25f, -MAX_VEL * 0.1f,
    -MAX_VEL * 0.05f, -MAX_VEL * 0.02f, 0.0f,
     MAX_VEL * 0.02f,  MAX_VEL * 0.05f,
     MAX_VEL * 0.1f,   MAX_VEL * 0.25f,  MAX_VEL * 0.5f,  MAX_VEL,
};
constexpr int ACTION_ZERO_IDX = 6;   // ACTIONS[6] == 0.0

// ACTION_DELAY = 2 (train_obs9_baseline.py:211): the command the simulator applied was the one
// chosen two control steps earlier, modelling MCU latency. The policy learned
// to compensate for that lag, so the firmware has to reproduce it — removing a
// delay the model expects makes it overcorrect.
constexpr int ACTION_DELAY = 2;

// Observation normalisation limits (train_obs9_baseline.py). Every obs element is value/limit
// clamped to [-1, 1].
constexpr float PITCH_LIMIT  = 1.2f;    // rad
constexpr float RATE_LIMIT   = 10.0f;   // rad/s
constexpr float VEL_LIMIT    = MAX_VEL; // rad/s
constexpr float LIN_VEL_LIM  = 0.19f;   // m/s (0.030 x 5.7 = 0.171 top)
constexpr float DRIFT_LIMIT  = 2.0f;    // m
constexpr float WHEEL_RADIUS = 0.030f;  // m

// Complementary filter (imu_filter.py).
constexpr float FILTER_ALPHA          = 0.995f;
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
#define I2C_FREQ_HZ   400000

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

// Encoder direction. Positive must mean "this wheel drives the robot forward".
// Check in MODE_DIAG: push the robot forward by hand, both wheel velocities
// should read positive. Flip a sign here if one reads negative.
constexpr float ENC1_SIGN = 1.0f;
constexpr float ENC2_SIGN = 1.0f;

// Velocity smoothing. At 200 Hz the slowest command (0.114 rad/s) is only ~0.4
// counts per tick, so the raw per-tick difference is very coarse at low speed.
// train.py injected VEL_NOISE = 0.02 rad/s during training, so the policy has
// some tolerance for this, but a light EMA still helps. 1.0 = no filtering.
constexpr float ENC_VEL_EMA_ALPHA = 0.3f;

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

// Motor direction. Positive duty must drive the robot forward.
constexpr float MOTOR_A_SIGN = 1.0f;
constexpr float MOTOR_B_SIGN = 1.0f;

// ── Per-wheel velocity controller ────────────────────────────────────
// The simulator drove the wheels with stiffness=0, damping=500 — a pure velocity
// servo with a 5 N.m torque cap. ACTIONS are therefore velocity *targets*, not
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
constexpr float VEL_KFF = 170.0f;   // PWM counts per (rad/s) of target

// Anti-windup, expressed as the largest CONTRIBUTION the integral term may make,
// in PWM counts. Clamping the raw integral instead would be meaningless — its
// useful range depends entirely on VEL_KI.
constexpr float VEL_I_CLAMP_PWM = 600.0f;

// Duty below which the motor does not turn at all. Measured, not guessed: in
// MODE_DIAG ramp the duty until the wheel starts moving.
constexpr int MOTOR_DEADBAND = 40;

// ── Safety ───────────────────────────────────────────────────────────
// Past this tilt the robot has fallen and the policy cannot recover it; cutting
// the motors stops it driving itself across the floor on its side.
constexpr float ABORT_PITCH_RAD = 0.7f;   // ~40 deg

// Motors are cut if a control tick takes longer than this — a stalled inference
// or a blocked I2C read must not leave the last command running.
constexpr uint32_t WATCHDOG_US = 50000;   // 50 ms
