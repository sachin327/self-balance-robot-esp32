// main.cpp — self-balancing robot policy firmware.
//
// Model contract (train_obs9_baseline.py, model_ep250):
//   input  (1, 9)  float32, every element normalised then clamped to [-1, 1]
//   output (1, 13) float32 Q-values; argmax indexes ACTIONS, a wheel-velocity
//                  command in rad/s applied to BOTH wheels
//   MAX_VEL 5.7 rad/s = 0.171 m/s at the 30 mm wheel — the 8 V pack ceiling
//
// Three modes, selected by ROBOT_MODE. It ships in MODE_DIAG on purpose: a
// flipped encoder sign or a swapped IMU axis is far cheaper to find with the
// wheels in the air than with the robot throwing itself at the floor.
//
//   MODE_DIAG      prints the 9 observation values, motors OFF. START HERE.
//   MODE_VALIDATE  50-vector model check against PyTorch y_true.
//   MODE_RUN       closed loop: observe -> infer -> argmax -> drive both wheels.
//
// MODE_RUN is gated on inference actually fitting the 5 ms budget. The previous
// 823 KB float32 model measured 119.8 ms per inference on this board because the
// weights stream from flash at ~6.9 MB/s. If that has not changed, RUN will
// report an overrun every tick and cannot balance the robot — the control path
// is correct, but it needs a model that meets the deadline.

#include <Arduino.h>
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
#endif

namespace {

Imu      g_imu;
Encoders g_enc;
Motors   g_motors;
Policy   g_policy;

bool     g_hw_ok   = false;
bool     g_running = false;   // MODE_RUN stays disarmed until 'g'
uint32_t g_overruns = 0, g_ticks = 0;

const char* kObsNames[OBS_DIM_CFG] = {
    "pitch", "p_rate", "roll", "r_rate",
    "w1", "w2", "vel_est", "x_disp", "app_cmd",
};

constexpr float RAD2DEG = 57.29577951f;

}  // namespace

static bool initHardware(bool need_model) {
  if (!g_imu.begin()) return false;

  Serial.println("hold the robot still and upright — calibrating gyro");
  delay(1500);
  g_imu.calibrateGyro(1000);

  ImuReading r;
  if (g_imu.read(r)) g_imu.update(r, CONTROL_DT_S, /*init=*/true);

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
  Serial.println("  c  zero encoder counts        r  print raw counts");
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
    case 'c': g_enc.resetCounts(); Serial.println("-> counts zeroed"); break;
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
      ImuReading r;
      if (g_imu.read(r)) g_imu.update(r, CONTROL_DT_S, /*init=*/true);
      Serial.println("-> odometry zeroed, filter re-initialised");
      break;
    }
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

  g_imu.update(r, CONTROL_DT_S);
  g_enc.update(CONTROL_DT_S);

  // Motors stay off; this only assembles and displays the observation.
  g_policy.buildObs(g_imu.pitch(), g_imu.pitchRate(),
                    g_imu.roll(), g_imu.rollRate(),
                    g_enc.vel1(), g_enc.vel2(), CONTROL_DT_S);

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

// Cosine after removing each row's mean. Raw cosine is near-useless here: the
// dueling value term is a large offset common to all 13 entries and pins it at
// 1.0 even when the advantage ranking is wrong.
static double centeredCosine(const float* a, const float* b, int n) {
  double ma = 0.0, mb = 0.0;
  for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < n; i++) {
    const double ca = a[i] - ma, cb = b[i] - mb;
    dot += ca * cb; na += ca * ca; nb += cb * cb;
  }
  const double d = sqrt(na) * sqrt(nb);
  return d > 0.0 ? dot / d : 0.0;
}

static void runValidation() {
  double sum_cent = 0.0;
  float worst = 0.0f;
  int matches = 0;
  uint32_t total_us = 0;

  Serial.println("\n  #   cosine(cent)   max|diff|   got ref      us");
  for (int v = 0; v < NUM_TEST_VECTORS; v++) {
    g_policy.setObs(kTestInputs[v]);      // fixture goes in verbatim
    const int a_got = g_policy.inferActionIdx();
    total_us += g_policy.lastInferenceUs();

    const float* got = g_policy.qValues();
    const float* ref = kRefOutputs[v];

    float md = 0.0f;
    int a_ref = 0;
    for (int i = 0; i < N_ACTIONS; i++) {
      md = fmaxf(md, fabsf(got[i] - ref[i]));
      if (ref[i] > ref[a_ref]) a_ref = i;
    }
    worst = fmaxf(worst, md);
    if (a_got == a_ref) matches++;

    const double cc = centeredCosine(got, ref, N_ACTIONS);
    sum_cent += cc;

    Serial.printf("  %2d   %.9f   %.3e   %2d  %2d %s %6lu\n",
                  v, cc, md, a_got, a_ref, (a_got == a_ref) ? " " : "X",
                  (unsigned long)g_policy.lastInferenceUs());
  }

  const uint32_t mean_us = total_us / NUM_TEST_VECTORS;
  Serial.println("\n  ===================== SUMMARY =====================");
  Serial.printf("  mean cosine (centered)  %.9f\n", sum_cent / NUM_TEST_VECTORS);
  Serial.printf("  max |diff|              %.6e\n", worst);
  Serial.printf("  argmax agreement        %d/%d\n", matches, NUM_TEST_VECTORS);
  Serial.printf("  inference               %lu us mean\n", (unsigned long)mean_us);
  Serial.printf("  200 Hz budget 5000 us   -> %s\n",
                (mean_us < CONTROL_DT_US) ? "FITS" : "TOO SLOW");
  Serial.println("  ===================================================");
}
#endif  // MODE_VALIDATE

// =====================================================================
#if ROBOT_MODE == MODE_RUN

static void runLoop() {
  static uint32_t next_tick = 0;
  static uint32_t next_print = 0;

  const uint32_t now = micros();
  if ((int32_t)(now - next_tick) < 0) return;
  const uint32_t tick_start = now;
  next_tick = now + CONTROL_DT_US;

  ImuReading r;
  if (!g_imu.read(r)) { g_motors.stop(); return; }

  g_imu.update(r, CONTROL_DT_S);
  g_enc.update(CONTROL_DT_S);

  // Fallen: the policy has no recovery behaviour and would keep driving.
  if (fabsf(g_imu.pitch()) > ABORT_PITCH_RAD) {
    if (g_running) {
      Serial.printf("FALLEN (%+.1f deg) — motors cut, 'g' to re-arm\n",
                    g_imu.pitch() * RAD2DEG);
      g_running = false;
    }
    g_motors.stop();
    g_policy.reset();
    return;
  }

  if (!g_running) { g_motors.stop(); return; }

  g_policy.buildObs(g_imu.pitch(), g_imu.pitchRate(),
                    g_imu.roll(), g_imu.rollRate(),
                    g_enc.vel1(), g_enc.vel2(), CONTROL_DT_S);

  const int chosen = g_policy.inferActionIdx();
  if (chosen < 0) {
    Serial.println("Invoke() failed — motors cut");
    g_running = false;
    g_motors.stop();
    return;
  }

  // Same command to both wheels, exactly as apply_action() does in training.
  const int   applied = g_policy.applyDelay(chosen);
  const float target  = ACTIONS[applied];
  g_motors.setVelocity(target, target, g_enc.vel1(), g_enc.vel2(), CONTROL_DT_S);

  const uint32_t elapsed = micros() - tick_start;
  g_ticks++;
  if (elapsed > CONTROL_DT_US) {
    g_overruns++;
    if ((g_overruns % 200) == 1) {
      Serial.printf("OVERRUN %lu/%lu — %lu us used of %lu\n",
                    (unsigned long)g_overruns, (unsigned long)g_ticks,
                    (unsigned long)elapsed, (unsigned long)CONTROL_DT_US);
    }
  }
  if (elapsed > WATCHDOG_US) {
    Serial.println("WATCHDOG — tick far over budget, motors cut");
    g_running = false;
    g_motors.stop();
    return;
  }

  if ((int32_t)(millis() - next_print) >= 0) {
    next_print = millis() + 250;
    Serial.printf("pitch %+6.1f deg  act %2d -> %+6.3f rad/s  "
                  "w %+5.2f/%+5.2f  duty %+5d/%+5d  %lu us\n",
                  g_imu.pitch() * RAD2DEG, applied, target,
                  g_enc.vel1(), g_enc.vel2(),
                  g_motors.duty1(), g_motors.duty2(),
                  (unsigned long)g_policy.lastInferenceUs());
  }
}
#endif  // MODE_RUN

// =====================================================================

void setup() {
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
  Serial.printf(" obs %d -> actions %d   MAX_VEL %.2f rad/s   %d Hz\n",
                OBS_DIM_CFG, N_ACTIONS_CFG, MAX_VEL,
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
  Serial.println("\nWheels OFF THE GROUND for the first arm.");
  Serial.println("'g' to arm, 's' to stop.");
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
    const char cmd = (char)Serial.read();
    if (cmd == 'g') {
      g_policy.reset();
      g_enc.resetCounts();
      g_overruns = 0; g_ticks = 0;
      g_running = true;
      Serial.println("ARMED");
    } else if (cmd == 's') {
      g_running = false;
      g_motors.stop();
      Serial.println("STOPPED");
    }
  }
  runLoop();
#endif
}
