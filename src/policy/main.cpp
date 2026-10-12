// main.cpp — self-balancing robot policy firmware.
//
// Model contract (PPO policy, 10 -> 64 -> 64 -> 1, int8 with float32 I/O):
//   input  (1, 10) float32, every element normalised then clamped to [-1, 1]
//   output (1, 1) float32 in [-1, 1], clipped inside the graph
//   wheel command = output * MAX_VEL (5.7 rad/s), the same for BOTH wheels,
//                   applied ACTION_DELAY ticks after it was computed
//
// QUANTIZE/DEQUANTIZE sit inside the graph, so this firmware never touches a
// scale or zero point.
//
// Three modes, selected by ROBOT_MODE (set per build environment):
//
//   MODE_DIAG      prints the 10 observation values, motors OFF.
//   MODE_VALIDATE  50-vector model check against the PyTorch reference, plus
//                  measured inference time and a per-op breakdown.
//   MODE_RUN       closed loop: observe -> infer -> delay -> drive both wheels.

#include <Arduino.h>
#include <Preferences.h>
#include <Wire.h>
#include <math.h>

#include "robot_config.h"
#include "imu.h"
#include "encoders.h"
#include "motors.h"
#include "policy.h"

#define MODE_DIAG     0
#define MODE_VALIDATE 1
#define MODE_RUN      2

#ifndef ROBOT_MODE
#define ROBOT_MODE MODE_DIAG
#endif

#if ROBOT_MODE == MODE_VALIDATE
#include "test_vectors.h"
// A fixture from a different model compiles fine and then reads past the end
// of each test row (setObs copies OBS_DIM_CFG values). Refuse it here instead.
static_assert(OBS_DIM == OBS_DIM_CFG,
              "include/test_vectors.h is for a different input count than "
              "OBS_DIM_CFG: copy the fixture that belongs to the linked model");
static_assert(N_OUTPUTS == N_OUTPUTS_CFG,
              "include/test_vectors.h is for a different output count than "
              "N_OUTPUTS_CFG");
#endif

namespace {

Imu      g_imu;
Encoders g_enc;
Motors   g_motors;
Policy   g_policy;

bool     g_hw_ok   = false;
bool     g_running = false;   // motors armed (MODE_RUN)
uint32_t g_overruns = 0, g_ticks = 0;

// Auto-arm: once the robot is upright, arm when millis() reaches g_arm_at_ms.
bool     g_arm_pending = false;
uint32_t g_arm_at_ms   = 0;

// False until the filter has been seeded from a still accelerometer reading.
// While false, every tick retries the seed and nothing else runs on the
// filter, so the auto-arm countdown starts from a settled tilt estimate.
bool     g_filter_seeded = false;

// Tilt demo ('t' in MODE_RUN): the policy sees tilt only, so in the hand the
// wheels follow the tilt both ways instead of running away. For checking the
// tilt response and the drive direction. NOT balancing: never use on the floor.
bool     g_tilt_demo = false;

// 'k' in MODE_RUN: ticks left to average the tilt for the balance trim, and
// the running sum. 0 = not measuring. The first TRIM_SETTLE_TICKS are skipped:
// after the previous trim step the policy is still walking the robot back to
// its start point, and the mean tilt is biased while it does.
constexpr int TRIM_SETTLE_TICKS = 600;    // 3 s
constexpr int TRIM_AVG_TICKS    = 2000;   // 10 s
int      g_trim_ticks_left = 0;
double   g_trim_sum = 0.0;

const char* kObsNames[OBS_DIM_CFG] = {
    "pitch", "p_rate", "roll", "r_rate",
    "w1", "w2", "vel_est", "x_disp", "app_cmd", "x_int",
};

constexpr float RAD2DEG = 57.29577951f;

}  // namespace

// Try the gated accel-only seed (see FILTER_SEED_MIN_TRUST). Prints once when
// it succeeds, so the boot log shows the tilt the policy will start from.
static bool trySeedFilter(const ImuReading& r) {
  if (g_filter_seeded) return true;
  g_filter_seeded = g_imu.trySeed(r, FILTER_SEED_MIN_TRUST);
  if (g_filter_seeded) {
    Serial.printf("IMU: filter seeded at pitch %+.2f deg, roll %+.2f deg\n",
                  g_imu.pitch() * RAD2DEG, g_imu.roll() * RAD2DEG);
  }
  return g_filter_seeded;
}

// Gyro bias lives in flash (NVS), so it is measured once, not on every boot.
// Repeated calibrations agree to ~0.003 rad/s, but the bias keeps moving after
// calibrating (warm-up): ~0.02 rad/s residual was measured at rest right after
// a fresh 'c'. It shifts the tilt estimate (~1 deg) and the velocity estimate;
// position (obs[7]) comes from encoder counts and is not affected.
static bool loadGyroBias() {
  Preferences prefs;
  if (!prefs.begin("imu", /*readOnly=*/true)) return false;   // never saved
  // The bias is only valid for the read conditions it was measured under.
  const bool ok = prefs.isKey("bx") && prefs.isKey("by") && prefs.isKey("bz") &&
                  prefs.isKey("cv") && prefs.getUInt("cv") == GYRO_CAL_VERSION;
  if (ok) {
    g_imu.setGyroBias(prefs.getFloat("bx"), prefs.getFloat("by"),
                      prefs.getFloat("bz"));
  }
  prefs.end();
  return ok;
}

// IMU mounting offset, in the same NVS namespace as the gyro bias. Absent means
// zero, i.e. no correction, which is how every build behaved before this.
static void loadMountOffset() {
  Preferences prefs;
  if (!prefs.begin("imu", /*readOnly=*/true)) return;
  // isKey first: getFloat on a missing key logs an NVS error line at boot.
  if (prefs.isKey("mp") && prefs.isKey("mr")) {
    const float mp = prefs.getFloat("mp"), mr = prefs.getFloat("mr");
    if (fabsf(mp) <= MOUNT_OFFSET_MAX_RAD && fabsf(mr) <= MOUNT_OFFSET_MAX_RAD) {
      g_imu.setMountOffset(mp, mr);
    } else {
      Serial.printf("IMU: saved mount offset %+.2f / %+.2f deg is over %.0f deg, "
                    "IGNORED (measured with the chassis not vertical?). "
                    "Redo 'o' or clear with 'O'.\n",
                    mp * RAD2DEG, mr * RAD2DEG, MOUNT_OFFSET_MAX_RAD * RAD2DEG);
    }
  }
  prefs.end();
}

// Balance trim, same NVS namespace. Absent means zero.
static void loadBalanceTrim() {
  Preferences prefs;
  if (!prefs.begin("imu", /*readOnly=*/true)) return;
  if (prefs.isKey("bt")) {
    const float bt = prefs.getFloat("bt");
    if (fabsf(bt) <= BALANCE_TRIM_MAX_RAD) g_imu.setBalanceTrim(bt);
  }
  prefs.end();
}

static void saveBalanceTrim(float pitch_rad) {
  g_imu.setBalanceTrim(pitch_rad);
  Preferences prefs;
  if (prefs.begin("imu", /*readOnly=*/false)) {
    prefs.putFloat("bt", pitch_rad);
    prefs.end();
  }
}

static void saveMountOffset(float pitch_rad, float roll_rad) {
  g_imu.setMountOffset(pitch_rad, roll_rad);
  Preferences prefs;
  if (prefs.begin("imu", /*readOnly=*/false)) {
    prefs.putFloat("mp", pitch_rad);
    prefs.putFloat("mr", roll_rad);
    prefs.end();
  }
}

// The robot must be still. Blocks for about 2 s.
static void calibrateAndSaveGyro() {
  Serial.println("calibrating gyro — keep the robot still (put it down, hands off)");
  delay(1500);
  // Keep trying until a 2 s window with the robot still. A calibration taken
  // while it is being handled would be worse than none.
  while (!g_imu.calibrateGyro()) {
    Serial.println("IMU: trying again — put the robot down and let go");
    delay(1000);
  }

  Preferences prefs;
  if (prefs.begin("imu", /*readOnly=*/false)) {
    prefs.putFloat("bx", g_imu.gyroBiasX());
    prefs.putFloat("by", g_imu.gyroBiasY());
    prefs.putFloat("bz", g_imu.gyroBiasZ());
    prefs.putUInt("cv", GYRO_CAL_VERSION);
    prefs.end();
    Serial.println("IMU: gyro calibration saved to flash");
  } else {
    Serial.println("IMU: could not save gyro calibration");
  }

  // Re-seed the filter: the robot was still for the calibration, so the next
  // tick's seed is accepted immediately, and in MODE_RUN that restarts the
  // 1 s arm countdown from the fresh seed.
  g_filter_seeded = false;
  ImuReading r;
  if (g_imu.read(r)) trySeedFilter(r);
}

static bool initHardware(bool need_model) {
  if (!g_imu.begin()) return false;

  if (loadGyroBias()) {
    Serial.printf("IMU: gyro bias from flash %.5f %.5f %.5f rad/s  ('c' to recalibrate)\n",
                  g_imu.gyroBiasX(), g_imu.gyroBiasY(), g_imu.gyroBiasZ());
  } else {
    // Never calibrated, or calibrated under different read conditions. Start
    // from zero so the new measurement doesn't refine a bias that isn't valid.
    g_imu.setGyroBias(0.0f, 0.0f, 0.0f);
    Serial.println("IMU: no gyro calibration for this firmware's read settings");
    calibrateAndSaveGyro();
  }

  loadMountOffset();
  Serial.printf("IMU: mount offset pitch %+.2f deg, roll %+.2f deg%s\n",
                g_imu.mountPitch() * RAD2DEG, g_imu.mountRoll() * RAD2DEG,
                (g_imu.mountPitch() == 0.0f && g_imu.mountRoll() == 0.0f)
                    ? "  (not measured, 'o' in policy_quick_check)" : "");

  loadBalanceTrim();
  Serial.printf("IMU: balance trim %+.2f deg%s\n", g_imu.balanceTrim() * RAD2DEG,
                g_imu.balanceTrim() == 0.0f ? "  (none; 'B' in policy_quick_check if it falls one way)" : "");

  // One seed attempt now. If the robot is being handled (trust too low), the
  // control loop keeps retrying every tick until it is still.
  ImuReading r;
  if (g_imu.read(r) && !trySeedFilter(r)) {
    Serial.println("IMU: robot moving, filter seed waits until it is still");
  }

  g_enc.begin();
  g_motors.begin();
  g_motors.stop();

  if (need_model && !g_policy.begin()) return false;

  Serial.printf("free heap %u bytes\n", (unsigned)ESP.getFreeHeap());
  return true;
}

// =====================================================================
#if ROBOT_MODE == MODE_DIAG

static void diagHelp() {
  Serial.println();
  Serial.println("  c  calibrate gyro + save      e  zero encoder counts");
  Serial.println("  r  print raw counts           p  step response 0 -> 2.85 rad/s (CSV)");
  Serial.println("                                P  step response 0 -> 0.20 rad/s (CSV)");
  Serial.println("  b  gear backlash (HOLD BOTH WHEELS so they cannot turn)");
  Serial.println("  g  gyro offset check: read timing and I2C speed (robot still, 10 s)");
  Serial.println("  G  gyro offset with MOTORS RUNNING (robot lying still, wheels free, 12 s)");
  Serial.println("  B  save balance point: hold the robot where it balances by itself");
  Serial.println("  N  clear balance point");
  Serial.println("  o  measure + save IMU mount offset (chassis held exactly vertical)");
  Serial.println("  O  clear IMU mount offset");
  Serial.println("  1  drive LEFT  wheel 2 s      2  drive RIGHT wheel 2 s");
  Serial.println("  d  deadband ramp              z  re-zero odometry + filter");
  Serial.println("  s  stop motors                h  help");
  Serial.println();
  Serial.println("  CHECK, wheels off the ground:");
  Serial.println("   - tilt forward   -> pitch rises positive, roll stays put");
  Serial.printf( "   - one wheel rev  -> count == %.0f (ENCODER_COUNTS_PER_REV)\n",
                 ENCODER_COUNTS_PER_REV);
  Serial.println("   - push forward   -> w1 AND w2 both positive");
  Serial.println("   - press '1'      -> LEFT drives forward, w1 positive");
  Serial.printf( "   - full duty tops out near %.2f rad/s (MAX_VEL)\n", MAX_VEL);
  Serial.println();
}

static void diagCommand(char cmd) {
  switch (cmd) {
    case 'c': calibrateAndSaveGyro(); break;
    case 'e':
      // x_disp is measured from the counts at the last policy reset, so reset
      // that reference too or the displayed distance jumps.
      g_enc.resetCounts(); g_policy.reset();
      Serial.println("-> counts zeroed");
      break;
    case 'r':
      Serial.printf("-> enc1(L)=%ld  enc2(R)=%ld   one rev = %.0f\n",
                    (long)g_enc.count1(), (long)g_enc.count2(),
                    ENCODER_COUNTS_PER_REV);
      break;
    case '1':
      Serial.println("-> LEFT forward 2 s");
      g_motors.setRawDuty(MOTOR_DEADBAND + 200, 0); delay(2000); g_motors.stop();
      break;
    case '2':
      Serial.println("-> RIGHT forward 2 s");
      g_motors.setRawDuty(0, MOTOR_DEADBAND + 200); delay(2000); g_motors.stop();
      break;
    case 'd':
      Serial.println("-> duty ramp; note where each wheel starts turning");
      for (int duty = 0; duty <= MOTOR_PWM_MAX; duty += 25) {
        g_motors.setRawDuty(duty, duty);
        delay(150);
        g_enc.update(0.15f);
        Serial.printf("   duty %4d   w1 %+6.2f   w2 %+6.2f rad/s\n",
                      duty, g_enc.vel1(), g_enc.vel2());
        if (Serial.available()) break;
      }
      g_motors.stop();
      break;
    case 'z': {
      g_policy.reset(); g_enc.resetCounts();
      g_filter_seeded = false;   // diagLoop re-seeds once the robot is still
      Serial.println("-> odometry zeroed, filter re-seeds when still");
      break;
    }
    case 'p':
    case 'P': {
      // Velocity-servo step response, for fitting the actuator lag in training.
      // The simulator drives an ideal velocity actuator (stiffness 0, damping
      // 500) that reaches the commanded speed within a tick; this PI loop plus
      // feedforward does not, and that difference is part of the real plant.
      //
      // Samples are buffered, not printed inside the loop: one Serial line at
      // 115200 baud takes ~4 ms and would wreck the 5 ms tick being measured.
      constexpr int kSteps = 60;
      static float  w1[kSteps], w2[kSteps];
      static int    d1[kSteps], d2[kSteps];
      static uint32_t ts[kSteps];

      // 'p': 2.85 rad/s, half of full command. 'P': 0.2 rad/s, a small step
      // that shows the deadband and how the servo handles near-zero commands.
      const float target = (cmd == 'p') ? MAX_VEL * 0.5f : 0.2f;
      Serial.printf("-> step 0 -> %.3f rad/s, %d ticks at %lu us (wheels in the air, or on the floor held upright)\n",
                    target, kSteps, (unsigned long)CONTROL_DT_US);
      g_motors.stop();
      g_enc.resetCounts();
      delay(300);                 // let the encoder window flush to zero

      const uint32_t t0 = micros();
      uint32_t next = t0;
      for (int k = 0; k < kSteps; k++) {
        while ((int32_t)(micros() - next) < 0) {}
        next += CONTROL_DT_US;
        g_enc.update(CONTROL_DT_S);
        g_motors.setVelocity(target, target, g_enc.vel1(), g_enc.vel2(),
                             CONTROL_DT_S);
        ts[k] = micros() - t0;
        w1[k] = g_enc.vel1(); w2[k] = g_enc.vel2();
        d1[k] = g_motors.duty1(); d2[k] = g_motors.duty2();
      }
      g_motors.stop();

      Serial.printf("# target %.4f rad/s  KFF %.0f KP %.0f KI %.0f  deadband %d\n",
                    target, VEL_KFF, VEL_KP, VEL_KI, MOTOR_DEADBAND);
      Serial.println("us,w1,w2,duty1,duty2");
      for (int k = 0; k < kSteps; k++) {
        Serial.printf("%lu,%.4f,%.4f,%d,%d\n", (unsigned long)ts[k],
                      w1[k], w2[k], d1[k], d2[k]);
      }
      Serial.println("# end");
      g_enc.resetCounts(); g_policy.reset();   // the test moved the wheels
      break;
    }
    case 'g': {
      // Gyro offset under different read conditions, robot still. After a
      // fresh 'c' the control loop still sees a steady non-zero rate, which
      // calibration by construction does not, so the sensor must read
      // differently in the two. This separates the candidates: read timing
      // (calibration reads every ~2 ms with delay(), the loop every 5 ms after
      // a busy-wait) and I2C speed (1 MHz is beyond the sensor's 400 kHz
      // rating). Values are what read() returns, i.e. after the saved bias.
      const float lsb_per_rads = GYRO_LSB_PER_DPS * RAD2DEG;
      auto measure = [&](const char* label, bool loop_timing) {
        double sx = 0, sy = 0, sz = 0; int n = 0, fails = 0;
        uint32_t next = micros();
        for (int k = 0; k < 400; k++) {
          if (loop_timing) {
            while ((int32_t)(micros() - next) < 0) {}
            next += CONTROL_DT_US;
          } else {
            delay(2);
          }
          ImuReading r;
          if (g_imu.read(r)) { sx += r.wx; sy += r.wy; sz += r.wz; n++; } else { fails++; }
        }
        if (n == 0) { Serial.printf("   %-34s no readings\n", label); return; }
        Serial.printf("   %-34s wx %+.4f  wy %+.4f  wz %+.4f rad/s   (%+4.0f %+4.0f %+4.0f LSB)  fails %d\n",
                      label, sx / n, sy / n, sz / n,
                      sx / n * lsb_per_rads, sy / n * lsb_per_rads, sz / n * lsb_per_rads, fails);
      };
      Serial.println("-> gyro offset check: keep the robot STILL for 10 s");
      delay(1000);
      Serial.printf("   I2C at %lu Hz:\n", (unsigned long)I2C_FREQ_HZ);
      measure("calibration timing (2 ms, delay)", false);
      measure("control-loop timing (5 ms, spin)", true);
      Wire.setClock(400000);
      delay(50);
      Serial.println("   I2C at 400000 Hz:");
      measure("calibration timing (2 ms, delay)", false);
      measure("control-loop timing (5 ms, spin)", true);
      Wire.setClock(I2C_FREQ_HZ);
      delay(50);
      Serial.println("   All four lines should be near zero and agree. wy feeds pitch, wx roll.");
      g_filter_seeded = false;   // the filter was not updated meanwhile
      break;
    }
    case 'G': {
      // Does the gyro's zero offset move when the motors draw current? The
      // offset follows bus/CPU activity ('g'), which points at the sensor's
      // supply, and the motors load the same battery. The robot must rest
      // still with the wheels free to spin (lying on its back or side), NOT
      // hand-held: any real rotation would be read as offset. Motor vibration
      // adds noise but should not move the mean.
      const float lsb_per_rads = GYRO_LSB_PER_DPS * RAD2DEG;
      auto measure = [&](const char* label, int duty) {
        g_motors.setRawDuty(duty, duty);
        delay(700);                                   // reach steady speed
        double sx = 0, sy = 0, sz = 0; int n = 0;
        uint32_t next = micros();
        for (int k = 0; k < 400; k++) {               // control-loop timing
          while ((int32_t)(micros() - next) < 0) {}
          next += CONTROL_DT_US;
          ImuReading r;
          if (g_imu.read(r)) { sx += r.wx; sy += r.wy; sz += r.wz; n++; }
        }
        g_motors.stop();
        if (n == 0) { Serial.printf("   %-22s no readings\n", label); return; }
        Serial.printf("   %-22s wx %+.4f  wy %+.4f  wz %+.4f rad/s   (%+4.0f %+4.0f %+4.0f LSB)\n",
                      label, sx / n, sy / n, sz / n,
                      sx / n * lsb_per_rads, sy / n * lsb_per_rads, sz / n * lsb_per_rads);
      };
      Serial.println("-> gyro offset vs motors: robot LYING STILL, wheels free to spin.");
      Serial.println("   Starting in 3 s...");
      delay(3000);
      measure("motors off", 0);
      measure("motors duty 300", 300);
      measure("motors duty 700", 700);
      measure("motors off again", 0);
      Serial.println("   If the motor lines differ from the off lines, the offset moves with motor load.");
      g_enc.resetCounts(); g_policy.reset();
      g_filter_seeded = false;
      break;
    }
    case 'b': {
      // Gear backlash. The encoder is on the motor shaft, behind the gearbox,
      // so with the WHEEL held still the motor can still turn through the gear
      // lash. Push gently one way, then the other, and the count swing between
      // the two rest positions is the lash, in counts already scaled to wheel
      // revolutions. A hand's give and tyre flex add to it, so this is an
      // upper bound; a swing far too large means the wheel itself turned.
      constexpr int kDuty   = MOTOR_DEADBAND + 80;   // gentle: ~12% of full duty
      constexpr int kCycles = 4;
      Serial.println("-> backlash: HOLD BOTH WHEELS firmly so they cannot turn.");
      Serial.println("   Starting in 3 s, gentle push each way, about 3 s total...");
      delay(3000);

      long swing1 = 0, swing2 = 0, max1 = 0, max2 = 0;
      for (int k = 0; k < kCycles; k++) {
        g_motors.setRawDuty(+kDuty, +kDuty); delay(350);
        const long a1 = g_enc.count1(), a2 = g_enc.count2();
        g_motors.setRawDuty(-kDuty, -kDuty); delay(350);
        const long b1 = g_enc.count1(), b2 = g_enc.count2();
        const long s1 = labs(a1 - b1), s2 = labs(a2 - b2);
        swing1 += s1; swing2 += s2;
        if (s1 > max1) max1 = s1;
        if (s2 > max2) max2 = s2;
        Serial.printf("   cycle %d: left %ld counts, right %ld counts\n", k + 1, s1, s2);
      }
      g_motors.stop();

      const float c1 = (float)swing1 / kCycles, c2 = (float)swing2 / kCycles;
      const float deg_per_count = 360.0f / ENCODER_COUNTS_PER_REV;
      const float mm_per_count  = WHEEL_RADIUS * 2.0f * PI / ENCODER_COUNTS_PER_REV * 1000.0f;
      Serial.printf("-> backlash  left  %.0f counts = %.2f deg = %.2f mm at the ground\n",
                    c1, c1 * deg_per_count, c1 * mm_per_count);
      Serial.printf("             right %.0f counts = %.2f deg = %.2f mm at the ground\n",
                    c2, c2 * deg_per_count, c2 * mm_per_count);
      if (max1 > 150 || max2 > 150) {
        Serial.println("   NOTE: over 150 counts (12 deg) in a cycle. A wheel probably");
        Serial.println("   turned in your hand; hold tighter and run 'b' again.");
      }
      g_enc.resetCounts(); g_policy.reset();   // the test moved the motors
      break;
    }
    case 'o': {
      // IMU mount offset: with the chassis held truly vertical and still, the
      // direction of gravity in the IMU's frame IS the board's rotation
      // relative to the chassis.
      //
      // Measured from the averaged ACCELEROMETER, not the filtered angle. A
      // gyro bias residual b shifts the complementary filter's steady state by
      // (1-c)/c * b * dt = ~0.995 s * b (c = 1 - FILTER_ALPHA); the measured
      // ~0.02 rad/s residual is ~1.1 deg. That part drifts with gyro warm-up
      // and isn't mounting, so it must not be baked into a fixed offset.
      // Rejected if the robot moved (accel trust dropped).
      Serial.println("-> mount offset: hold the chassis EXACTLY vertical and still");
      Serial.println("   (phone level against the flat side of the chassis, both axes).");
      Serial.println("   Measuring in 3 s...");
      delay(3000);

      double sax = 0.0, say = 0.0, saz = 0.0;
      float min_trust = 1.0f;
      int n = 0;
      uint32_t next = micros();
      for (int k = 0; k < 400; k++) {            // 400 x 5 ms = 2 s
        while ((int32_t)(micros() - next) < 0) {}
        next += CONTROL_DT_US;
        ImuReading r;
        if (!g_imu.read(r)) continue;
        min_trust = fminf(min_trust, Imu::accelTrust(r.ax, r.ay, r.az));
        sax += r.ax; say += r.ay; saz += r.az;
        n++;
      }
      if (n < 300 || min_trust < FILTER_SEED_MIN_TRUST) {
        Serial.printf("-> REJECTED: robot moved (min accel trust %.2f, %d samples). "
                      "Nothing saved, try again.\n", min_trust, n);
        break;
      }
      // Same angle definitions as Imu::update / trySeed.
      const float p  = atan2f(-(float)(sax / n), (float)(saz / n));
      const float rl = atan2f( (float)(say / n), (float)(saz / n));
      if (fabsf(p) > MOUNT_OFFSET_MAX_RAD || fabsf(rl) > MOUNT_OFFSET_MAX_RAD) {
        Serial.printf("-> REJECTED: pitch %+.2f deg, roll %+.2f deg is more than %.0f deg.\n"
                      "   The chassis was not vertical. Hold it upright against a phone\n"
                      "   level (both axes) and press 'o' again. Nothing saved.\n",
                      p * RAD2DEG, rl * RAD2DEG, MOUNT_OFFSET_MAX_RAD * RAD2DEG);
        break;
      }
      saveMountOffset(p, rl);
      Serial.printf("-> saved mount offset: pitch %+.2f deg, roll %+.2f deg\n",
                    p * RAD2DEG, rl * RAD2DEG);
      break;
    }
    case 'B': {
      // Balance point: the tilt at which the robot's weight is over the axle.
      // The user finds it by feel (wheels on the floor, two fingertips on top,
      // the pose where it has no tendency to tip either way) and holds it.
      // Measured from the accelerometer, like the mount offset, so gyro bias
      // doesn't enter. Stored relative to the mount offset.
      Serial.println("-> balance point: wheels on the floor, hold the top lightly with two");
      Serial.println("   fingers and find the lean where it does NOT want to tip either way.");
      Serial.println("   Hold it there. Measuring in 5 s...");
      delay(5000);
      double sax = 0.0, saz = 0.0;
      float min_trust = 1.0f;
      int n = 0;
      uint32_t next = micros();
      for (int k = 0; k < 400; k++) {
        while ((int32_t)(micros() - next) < 0) {}
        next += CONTROL_DT_US;
        ImuReading r;
        if (!g_imu.read(r)) continue;
        min_trust = fminf(min_trust, Imu::accelTrust(r.ax, r.ay, r.az));
        sax += r.ax; saz += r.az;
        n++;
      }
      if (n < 300 || min_trust < FILTER_SEED_MIN_TRUST) {
        Serial.printf("-> REJECTED: robot moved (min accel trust %.2f). Nothing saved, try again.\n",
                      min_trust);
        break;
      }
      const float trim = atan2f(-(float)(sax / n), (float)(saz / n)) - g_imu.mountPitch();
      if (fabsf(trim) > BALANCE_TRIM_MAX_RAD) {
        Serial.printf("-> REJECTED: %+.2f deg is more than %.0f deg. Nothing saved.\n",
                      trim * RAD2DEG, BALANCE_TRIM_MAX_RAD * RAD2DEG);
        break;
      }
      saveBalanceTrim(trim);
      Serial.printf("-> saved balance point: chassis leans %+.2f deg when balanced.\n"
                    "   The tilt shown now reads 0 at that pose.\n", trim * RAD2DEG);
      g_filter_seeded = false;
      break;
    }
    case 'N':
      saveBalanceTrim(0.0f);
      Serial.println("-> balance point cleared (0)");
      break;
    case 'O':
      saveMountOffset(0.0f, 0.0f);
      Serial.println("-> mount offset cleared (0, 0)");
      break;
    case 's': g_motors.stop(); Serial.println("-> stopped"); break;
    case 'h': diagHelp(); break;
    default: break;
  }
}

static void diagLoop() {
  static uint32_t next_tick = 0, next_print = 0;

  const uint32_t now = micros();
  if ((int32_t)(now - next_tick) < 0) return;
  next_tick = now + CONTROL_DT_US;

  ImuReading r;
  if (!g_imu.read(r)) return;

  // Same gated seed as MODE_RUN, so the printed observation matches what the
  // policy would see.
  if (!trySeedFilter(r)) return;

  g_imu.update(r, CONTROL_DT_S);
  g_enc.update(CONTROL_DT_S);

  // Motors stay off; this only assembles and displays the observation.
  g_policy.buildObs(g_imu.pitch(), g_imu.pitchRate(),
                    g_imu.roll(), g_imu.rollRate(),
                    g_enc.vel1(), g_enc.vel2(),
                    g_enc.count1(), g_enc.count2(), CONTROL_DT_S);

  // Print at 5 Hz while the filter and odometry keep integrating at 200 Hz.
  if ((int32_t)(millis() - next_print) >= 0) {
    next_print = millis() + 200;
    const float* o = g_policy.obs();
    Serial.print("obs ");
    for (int i = 0; i < OBS_DIM_CFG; i++) {
      Serial.printf("%s=%+.3f ", kObsNames[i], o[i]);
    }
    Serial.printf("| %+.1f/%+.1f deg  trust %.2f\n",
                  g_imu.pitch() * RAD2DEG, g_imu.roll() * RAD2DEG,
                  Imu::accelTrust(r.ax, r.ay, r.az));
  }
}
#endif  // MODE_DIAG

// =====================================================================
#if ROBOT_MODE == MODE_VALIDATE

// Pass limits for the int8 model against its PyTorch reference, in rad/s of
// wheel command. From the training side: on the fixture the host int8 model
// shows mean 0.033 and max 0.198 rad/s; one int8 output step is 0.045 rad/s.
constexpr float kValidateMaxErr  = 0.35f;
constexpr float kValidateMeanErr = 0.10f;

static void runValidation() {
  double sum_err = 0.0;
  float worst = 0.0f;
  uint32_t total_us = 0;
  bool all_ok = true;

  Serial.println("\n  wheel command in rad/s (network output x MAX_VEL)");
  Serial.println("  #      board        ref     |err|      us");
  for (int v = 0; v < NUM_TEST_VECTORS; v++) {
    g_policy.setObs(kTestInputs[v]);      // fixture goes in verbatim
    float got = 0.0f;
    if (!g_policy.inferCommand(&got)) { all_ok = false; continue; }
    total_us += g_policy.lastInferenceUs();

    const float ref = kRefOutputs[v][0] * MAX_VEL;
    const float err = fabsf(got - ref);
    sum_err += err;
    worst = fmaxf(worst, err);

    Serial.printf("  %2d   %+8.4f   %+8.4f   %7.4f   %6lu\n", v, got, ref, err,
                  (unsigned long)g_policy.lastInferenceUs());
  }

  const uint32_t mean_us = total_us / NUM_TEST_VECTORS;
  const float mean_err = (float)(sum_err / NUM_TEST_VECTORS);
  const bool pass = all_ok && worst <= kValidateMaxErr && mean_err <= kValidateMeanErr;
  Serial.println("\n  ===================== SUMMARY =====================");
  Serial.printf("  mean |error|            %.4f rad/s  (limit %.2f)\n", mean_err, kValidateMeanErr);
  Serial.printf("  max  |error|            %.4f rad/s  (limit %.2f)\n", worst, kValidateMaxErr);
  Serial.printf("  model vs reference      -> %s\n", pass ? "PASS" : "FAIL");
  Serial.printf("  inference               %lu us mean\n", (unsigned long)mean_us);
  Serial.printf("  200 Hz budget 5000 us   -> %s\n",
                (mean_us < CONTROL_DT_US) ? "FITS" : "TOO SLOW");
  Serial.println("  ===================================================");

  // Where the inference time goes, per op type. A separate pass so the
  // profiler's own micros() calls don't inflate the timings reported above.
  g_policy.setOpProfiling(true);
  for (int v = 0; v < NUM_TEST_VECTORS; v++) {
    float unused = 0.0f;
    g_policy.setObs(kTestInputs[v]);
    g_policy.inferCommand(&unused);
  }
  g_policy.setOpProfiling(false);
  g_policy.printOpProfile(NUM_TEST_VECTORS);
}
#endif  // MODE_VALIDATE

// =====================================================================
#if ROBOT_MODE == MODE_RUN

// Per-stage timing, so an overrun says WHICH stage is costing the budget
// rather than just that the tick was late. Six micros() calls per tick, about
// 1 us each; the summary prints once a second and only with DEBUG_PRINTS on.
struct StageStat {
  uint32_t sum = 0, max = 0, min = 0xFFFFFFFF, n = 0;
  void add(uint32_t us) {
    sum += us;
    if (us > max) max = us;
    if (us < min) min = us;
    n++;
  }
  uint32_t avg() const { return n ? sum / n : 0; }
  uint32_t lo() const { return n ? min : 0; }
  void reset() { sum = 0; max = 0; min = 0xFFFFFFFF; n = 0; }
};
static StageStat g_t_imu, g_t_enc, g_t_obs, g_t_infer, g_t_motor, g_t_tick;

static void runLoop() {
  static uint32_t next_tick = 0;
  static uint32_t next_print = 0;
  static uint32_t next_timing = 0;

  const uint32_t now = micros();
  if ((int32_t)(now - next_tick) < 0) return;
  const uint32_t tick_start = now;
  next_tick = now + CONTROL_DT_US;

  ImuReading r;
  if (!g_imu.read(r)) { g_motors.stop(); return; }
  const uint32_t t_after_imu = micros();

  // Not seeded yet (moving at power-on, or just recalibrated): retry the seed,
  // keep motors off and the arm countdown un-started. g_arm_pending=false
  // means the 1 s countdown begins on the tick AFTER the seed is accepted, so
  // the policy always starts from a settled filter.
  if (!g_filter_seeded) {
    g_motors.stop();
    g_running = false;
    g_arm_pending = false;
    trySeedFilter(r);
    return;
  }

  g_imu.update(r, CONTROL_DT_S);
  g_enc.update(CONTROL_DT_S);
  const uint32_t t_after_enc = micros();

  // Fallen: the policy has no recovery behaviour and would keep driving.
  // Unconditional — a safety event, not debug noise, and it fires once per
  // fall rather than every tick.
  if (fabsf(g_imu.pitch()) > ABORT_PITCH_RAD) {
    if (g_running) {
      Serial.printf("FALLEN (%+.1f deg) — motors cut, stand it up to restart\n",
                    g_imu.pitch() * RAD2DEG);
      g_running = false;
    }
    g_arm_pending = false;   // the 1 s restarts once it is back upright
    g_motors.stop();
    g_policy.reset();
    return;
  }

  // Auto-arm AUTO_ARM_DELAY_MS after power-on, after 'c', or after being stood
  // back up following a fall, but only while inside +/-ARM_PITCH_RAD. Leaving
  // the window restarts the countdown. No key press needed.
  if (!g_running) {
    g_motors.stop();
    static bool told_outside = false;   // print the hint once per excursion
    if (fabsf(g_imu.pitch()) > ARM_PITCH_RAD || fabsf(g_imu.roll()) > ARM_ROLL_RAD) {
      if (!told_outside) {
        Serial.printf("waiting: pitch %+.1f deg (needs +/-%.0f), roll %+.1f deg (needs +/-%.0f)\n",
                      g_imu.pitch() * RAD2DEG, ARM_PITCH_RAD * RAD2DEG,
                      g_imu.roll() * RAD2DEG, ARM_ROLL_RAD * RAD2DEG);
        told_outside = true;
      }
      g_arm_pending = false;
      return;
    }
    told_outside = false;
    if (!g_arm_pending) {
      g_arm_pending = true;
      g_arm_at_ms = millis() + AUTO_ARM_DELAY_MS;
    }
    if ((int32_t)(millis() - g_arm_at_ms) < 0) return;

    g_arm_pending = false;
    g_policy.reset();
    g_enc.resetCounts();
    g_overruns = 0; g_ticks = 0;
    g_running = true;
    Serial.println("ARMED");
    return;
  }

  g_policy.buildObs(g_imu.pitch(), g_imu.pitchRate(),
                    g_imu.roll(), g_imu.rollRate(),
                    g_enc.vel1(), g_enc.vel2(),
                    g_enc.count1(), g_enc.count2(), CONTROL_DT_S);
  const uint32_t t_after_obs = micros();

  if (g_trim_ticks_left > 0) {
    if (g_trim_ticks_left <= TRIM_AVG_TICKS) g_trim_sum += g_imu.pitch();
    if (--g_trim_ticks_left == 0) {
      const float mean = (float)(g_trim_sum / TRIM_AVG_TICKS);
      const float trim = g_imu.balanceTrim() + mean;
      if (fabsf(trim) <= BALANCE_TRIM_MAX_RAD) {
        saveBalanceTrim(trim);
        Serial.printf("balance trim: mean tilt was %+.2f deg, trim now %+.2f deg (saved)\n",
                      mean * RAD2DEG, trim * RAD2DEG);
      } else {
        Serial.printf("balance trim: %+.2f deg would exceed the limit, not saved\n",
                      trim * RAD2DEG);
      }
    }
  }

  if (g_tilt_demo) g_policy.zeroWheelObs();

  // Out of range: lifted with the wheels spinning in the air, or shoved too
  // far. Training ends the episode at DRIFT_LIMIT, so the policy has no
  // trained behaviour beyond it. Cut the motors like a fall; the auto-arm
  // above restarts it with a fresh start point once it is upright and still.
  static uint32_t far_since_ms = 0;
  static bool     far = false;
  if (!g_tilt_demo && fabsf(g_policy.xDisp()) >= DRIFT_LIMIT) {
    if (!far) { far = true; far_since_ms = millis(); }
    if (millis() - far_since_ms >= DRIFT_HOLD_MS) {
      Serial.printf("TOO FAR (%+.2f m from start, limit %.2f) — motors cut, restarting here\n",
                    g_policy.xDisp(), DRIFT_LIMIT);
      far = false;
      g_running = false;
      g_arm_pending = false;
      g_motors.stop();
      g_policy.reset();
      return;
    }
  } else {
    far = false;
  }

  float cmd = 0.0f;
  const bool infer_ok = g_policy.inferCommand(&cmd);
  const uint32_t t_after_infer = micros();
  if (!infer_ok) {
    Serial.println("Invoke() failed — motors cut");
    g_running = false;
    g_motors.stop();
    return;
  }

  // Same command to both wheels, as in training, delayed by ACTION_DELAY ticks.
  const float target = g_policy.applyDelay(cmd);
  g_motors.setVelocity(target, target, g_enc.vel1(), g_enc.vel2(), CONTROL_DT_S);

  const uint32_t elapsed = micros() - tick_start;
  g_t_imu.add(t_after_imu - tick_start);
  g_t_enc.add(t_after_enc - t_after_imu);
  g_t_obs.add(t_after_obs - t_after_enc);
  g_t_infer.add(t_after_infer - t_after_obs);
  g_t_motor.add(micros() - t_after_infer);
  g_t_tick.add(elapsed);
  g_ticks++;
  if (elapsed > CONTROL_DT_US) {
    g_overruns++;
    // Diagnostic, not safety — WATCHDOG below is what actually protects the
    // robot. Gated: printing here is itself slow and can extend the very
    // overrun it reports, or cause the next tick to overrun too.
    if ((g_overruns % 200) == 1) {
      DBG_PRINTF("OVERRUN %lu/%lu — %lu us used of %lu\n",
                 (unsigned long)g_overruns, (unsigned long)g_ticks,
                 (unsigned long)elapsed, (unsigned long)CONTROL_DT_US);
    }
  }
  if (elapsed > WATCHDOG_US) {
    // Unconditional — this is why the motors just cut.
    Serial.println("WATCHDOG — tick far over budget, motors cut");
    g_running = false;
    g_motors.stop();
    return;
  }

  if ((int32_t)(millis() - next_print) >= 0) {
    // 20 Hz, one short line: what the policy saw and what it asked for.
    //   pitch deg | vel m/s | x m (obs[7] before /0.3) | xi m*s (obs[9] before
    //   /0.5) | out = network output [-1,1] | app = command on the wheels,
    //   rad/s | w = measured wheel speeds | d = motor duty
    // About 95 characters, so 1,900 chars/s of the 11,520 the port carries;
    // the TX buffer set in setup() keeps the write from blocking the tick.
    next_print = millis() + 50;
    DBG_PRINTF("pitch %+6.2f vel %+6.3f x %+6.3f xi %+6.3f out %+6.3f app %+6.3f "
               "w %+5.2f/%+5.2f d %+5d/%+5d\n",
               g_imu.pitch() * RAD2DEG, g_policy.linVel(), g_policy.xDisp(),
               g_policy.xInt(), g_policy.rawOutput(), target,
               g_enc.vel1(), g_enc.vel2(),
               g_motors.duty1(), g_motors.duty2());
  }

  // Once a second: where the 5000 us went, as avg/max per stage.
  if ((int32_t)(millis() - next_timing) >= 0) {
    next_timing = millis() + 1000;
    DBG_PRINTF("timing us (avg/max)  imu %lu/%lu  enc+filt %lu/%lu  obs %lu/%lu  "
               "infer %lu/%lu  motor %lu/%lu  TICK %lu/%lu (min %lu)  over %lu/%lu\n",
               (unsigned long)g_t_imu.avg(),   (unsigned long)g_t_imu.max,
               (unsigned long)g_t_enc.avg(),   (unsigned long)g_t_enc.max,
               (unsigned long)g_t_obs.avg(),   (unsigned long)g_t_obs.max,
               (unsigned long)g_t_infer.avg(), (unsigned long)g_t_infer.max,
               (unsigned long)g_t_motor.avg(), (unsigned long)g_t_motor.max,
               (unsigned long)g_t_tick.avg(),  (unsigned long)g_t_tick.max,
               (unsigned long)g_t_tick.lo(),
               (unsigned long)g_overruns,      (unsigned long)g_ticks);
    g_t_imu.reset(); g_t_enc.reset(); g_t_obs.reset();
    g_t_infer.reset(); g_t_motor.reset(); g_t_tick.reset();
  }
}
#endif  // MODE_RUN

// =====================================================================

void setup() {
  // Room for the 20 Hz debug line (MODE_RUN with DEBUG_PRINTS) so a print
  // queues instead of waiting on the UART. Must be set before begin().
  Serial.setTxBufferSize(2048);
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n=================================================================");
#if ROBOT_MODE == MODE_DIAG
  Serial.println(" SELF-BALANCING ROBOT — MODE_DIAG (motors off)");
#elif ROBOT_MODE == MODE_VALIDATE
  Serial.println(" SELF-BALANCING ROBOT — MODE_VALIDATE");
#else
  Serial.println(" SELF-BALANCING ROBOT — MODE_RUN");
#endif
  Serial.printf(" obs %d -> outputs %d   MAX_VEL %.2f rad/s   %d Hz\n",
                OBS_DIM_CFG, N_OUTPUTS_CFG, MAX_VEL,
                (int)(1.0f / CONTROL_DT_S));
  Serial.println("=================================================================");

  g_hw_ok = initHardware(/*need_model=*/ROBOT_MODE != MODE_DIAG);
  if (!g_hw_ok) {
    Serial.println("hardware init FAILED — halting");
    return;
  }

#if ROBOT_MODE == MODE_DIAG
  diagHelp();
#elif ROBOT_MODE == MODE_VALIDATE
  runValidation();
#else
  Serial.printf("\nStarts by itself after %lu ms held still within +/-%.0f deg of upright.\n"
                "'c' recalibrates the gyro. 'k' refines the balance point while it balances.\n"
                "'t' toggles the tilt demo (in the hand only).\n",
                (unsigned long)AUTO_ARM_DELAY_MS, ARM_PITCH_RAD * RAD2DEG);
#endif
}

void loop() {
  if (!g_hw_ok) { delay(1000); return; }

#if ROBOT_MODE == MODE_DIAG
  if (Serial.available()) diagCommand((char)Serial.read());
  diagLoop();

#elif ROBOT_MODE == MODE_VALIDATE
  delay(10000);

#else
  if (Serial.available()) {
    const char key = (char)Serial.read();
    if (key == 'c') {
      // Motors off while measuring; the robot must be still. Auto-arms again
      // AUTO_ARM_DELAY_MS after this finishes.
      g_running = false;
      g_arm_pending = false;
      g_motors.stop();
      calibrateAndSaveGyro();
    } else if (key == 'k') {
      // Refine the balance trim from the robot's own balancing: when it holds
      // position, the mean tilt it reads IS the error in the trim (it can only
      // stand still with its weight over the axle). Waits 3 s, then averages 10 s.
      if (g_running) {
        g_trim_ticks_left = TRIM_SETTLE_TICKS + TRIM_AVG_TICKS;
        g_trim_sum = 0.0;
        Serial.println("balance trim: measuring for 13 s — let it balance, don't touch");
      } else {
        Serial.println("balance trim: only while it is balancing (ARMED)");
      }
    } else if (key == 't') {
      g_tilt_demo = !g_tilt_demo;
      g_running = false;          // re-arm cleanly in the new mode
      g_arm_pending = false;
      g_motors.stop();
      g_policy.reset();
      if (g_tilt_demo) {
        Serial.println("TILT DEMO ON — hold the robot IN YOUR HAND, wheels off the floor.");
        Serial.println("  Tip the top one way: the wheels must drive TOWARDS that side,");
        Serial.println("  faster the more you tip. Tip the other way: they reverse.");
        Serial.println("  About 0.85 rad/s per degree, full speed near 7 deg. 't' again to leave.");
      } else {
        Serial.println("TILT DEMO OFF — normal balancing");
      }
    }
  }
  runLoop();
#endif
}
