#include "policy.h"

#include <math.h>
#include <string.h>

#include <esp_heap_caps.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

// The 10 -> 64 -> 64 -> 1 int8 graph needs well under 2 KB; begin() prints the
// real figure.
constexpr int kArenaSize = 8 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];

// The whole graph: QUANTIZE, FULLY_CONNECTED, TANH, FULLY_CONNECTED, TANH,
// FULLY_CONNECTED (output clip to [-1, 1] fused in), DEQUANTIZE. All four ops
// have int8 kernels in this library. A resolver one entry short, or a model
// using any other op, fails inside AllocateTensors().
constexpr int kNumOps = 4;

tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// Per-operator timing. TFLM brackets every op's Invoke() with BeginEvent /
// EndEvent, tagged with the op name (a static string, so the pointer is a
// stable key). Ops run in the same order every inference, so the position
// within an inference is the graph node index.
class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxTags = 8;
  static constexpr int kMaxNodes = 16;
  struct Entry { const char* tag; uint64_t total_us; uint32_t calls; };

  uint32_t BeginEvent(const char* tag) override {
    if (!enabled_) return 0;
    tag_ = tag;
    start_ = micros();
    return 0;
  }
  void EndEvent(uint32_t) override {
    if (!enabled_ || tag_ == nullptr) return;
    const uint32_t us = micros() - start_;
    if (node_ < kMaxNodes) {
      node_tag_[node_] = tag_;
      node_us_[node_] += us;
    }
    node_++;
    for (int i = 0; i < n_; i++) {
      if (entries_[i].tag == tag_) { entries_[i].total_us += us; entries_[i].calls++; return; }
    }
    if (n_ < kMaxTags) entries_[n_++] = {tag_, us, 1};
  }

  void startInference() { node_ = 0; }
  void endInference() { if (enabled_ && node_ > nodes_seen_) nodes_seen_ = node_; }

  void setEnabled(bool on) {
    enabled_ = on;
    if (on) {   // each enable starts a fresh measurement
      n_ = 0; tag_ = nullptr; node_ = 0; nodes_seen_ = 0;
      for (int i = 0; i < kMaxNodes; i++) { node_us_[i] = 0; node_tag_[i] = nullptr; }
    }
  }

  int count() const { return n_; }
  const Entry& entry(int i) const { return entries_[i]; }
  int nodes() const { return nodes_seen_ < kMaxNodes ? nodes_seen_ : kMaxNodes; }
  const char* nodeTag(int i) const { return node_tag_[i]; }
  uint64_t nodeUs(int i) const { return node_us_[i]; }

 private:
  bool enabled_ = false;
  const char* tag_ = nullptr;
  uint32_t start_ = 0;
  Entry entries_[kMaxTags] = {};
  int n_ = 0;
  int node_ = 0, nodes_seen_ = 0;
  const char* node_tag_[kMaxNodes] = {};
  uint64_t node_us_[kMaxNodes] = {};
};
OpProfiler g_op_profiler;

// Model bytes the interpreter reads: g_model in flash, or a RAM copy of it
// when MODEL_IN_RAM (see robot_config.h for why).
const unsigned char* g_model_bytes = nullptr;

// Normalise then clamp to [-1, 1], as training does for every observation.
inline float c(float v, float limit) {
  const float x = v / limit;
  return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

}  // namespace

bool Policy::begin() {
  // This all runs once at boot, never in the 5 ms control loop. DBG_* lines are
  // debug info; plain Serial lines are failure reasons needed in every build.
  //
  // Print what is actually linked in BEFORE anything else can fail — this is
  // the check that catches "I copied in a new model but the old one is still
  // what got flashed".
  DBG_PRINTF("Policy: g_model_len = %u bytes (%.1f KB)\n",
             (unsigned)g_model_len, g_model_len / 1024.0f);

  g_model_bytes = g_model;
  if (MODEL_IN_RAM) {
    // 16-byte aligned: TFLM reads the flatbuffer in place and traps on an
    // unaligned access, same reason g_model itself is alignas(16).
    auto* ram = static_cast<unsigned char*>(heap_caps_aligned_alloc(
        16, g_model_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (ram != nullptr) {
      memcpy(ram, g_model, g_model_len);
      g_model_bytes = ram;
    } else {
      Serial.println("Policy: no RAM for the model copy, running it from flash");
    }
  }
  DBG_PRINTF("Policy: model runs from %s\n",
             g_model_bytes == g_model ? "FLASH" : "RAM");

  const tflite::Model* model = tflite::GetModel(g_model_bytes);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Policy: schema version %lu, expected %d\n",
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<kNumOps> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||   // the three Linear layers
      resolver.AddTanh()           != kTfLiteOk ||   // hidden activations
      resolver.AddQuantize()       != kTfLiteOk ||   // float32 input -> int8
      resolver.AddDequantize()     != kTfLiteOk) {   // int8 -> float32 output
    Serial.println("Policy: operator registration failed");
    return false;
  }

  // The profiler is attached permanently but records nothing until
  // setOpProfiling(true), so normal inference pays one early return per op.
  static tflite::MicroInterpreter interpreter(model, resolver, g_arena,
                                              kArenaSize, nullptr,
                                              &g_op_profiler);
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Policy: AllocateTensors() failed — arena too small, or the");
    Serial.println("        model uses an operator other than FullyConnected,");
    Serial.println("        Tanh, Quantize, Dequantize (e.g. a pre-PPO model).");
    return false;
  }

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Catch a model/firmware mismatch here rather than as silent garbage later.
  if (g_input->type != kTfLiteFloat32 || g_output->type != kTfLiteFloat32) {
    Serial.println("Policy: expected float32 input and output tensors");
    return false;
  }
  if (g_input->dims->data[1] != OBS_DIM_CFG) {
    Serial.printf("Policy: model wants %d inputs, firmware builds %d\n",
                  g_input->dims->data[1], OBS_DIM_CFG);
    return false;
  }
  if (g_output->dims->data[1] != N_OUTPUTS_CFG) {
    Serial.printf("Policy: model has %d outputs, firmware expects %d "
                  "(13 = a pre-PPO discrete-action model)\n",
                  g_output->dims->data[1], N_OUTPUTS_CFG);
    return false;
  }

  reset();
  Serial.printf("Policy: ready — %u -> %u, arena %u/%d bytes\n",
                (unsigned)g_input->dims->data[1],
                (unsigned)g_output->dims->data[1],
                (unsigned)g_interpreter->arena_used_bytes(), kArenaSize);
  return true;
}

void Policy::setOpProfiling(bool on) { g_op_profiler.setEnabled(on); }

void Policy::printOpProfile(int inferences) const {
  uint64_t total = 0;
  for (int i = 0; i < g_op_profiler.count(); i++) total += g_op_profiler.entry(i).total_us;
  if (inferences <= 0 || total == 0) {
    Serial.println("  (no profile recorded)");
    return;
  }
  Serial.printf("\n  per-op time over %d inferences, model in %s\n", inferences,
                g_model_bytes == g_model ? "FLASH" : "RAM");
  Serial.println("  op                 calls/inf   us/inf    share");
  for (int i = 0; i < g_op_profiler.count(); i++) {
    const auto& e = g_op_profiler.entry(i);
    Serial.printf("  %-18s %8.1f %8.0f   %5.1f%%\n", e.tag,
                  (double)e.calls / inferences, (double)e.total_us / inferences,
                  100.0 * (double)e.total_us / (double)total);
  }
  Serial.printf("  %-18s %8s %8.0f   100.0%%\n", "TOTAL (ops only)", "",
                (double)total / inferences);

  // Per graph node, in execution order (same order as the .tflite's operators).
  Serial.println("\n  node,op,us_per_inference");
  for (int i = 0; i < g_op_profiler.nodes(); i++) {
    Serial.printf("  %d,%s,%.1f\n", i,
                  g_op_profiler.nodeTag(i) ? g_op_profiler.nodeTag(i) : "?",
                  (double)g_op_profiler.nodeUs(i) / inferences);
  }
}

size_t Policy::arenaUsedBytes() const {
  return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}

void Policy::reset() {
  x_disp_m_ = 0.0f;
  x_int_ms_ = 0.0f;
  ref_latched_ = false;
  lin_vel_ms_ = 0.0f;
  for (int i = 0; i < ACTION_DELAY; i++) delay_buf_[i] = 0.0f;
  applied_cmd_ = 0.0f;
  raw_output_ = 0.0f;
  for (int i = 0; i < OBS_DIM_CFG; i++) obs_[i] = 0.0f;
}

void Policy::setObs(const float* obs) {
  for (int i = 0; i < OBS_DIM_CFG; i++) obs_[i] = obs[i];
}

void Policy::buildObs(float pitch, float pitch_rate,
                      float roll, float roll_rate,
                      float w1, float w2,
                      int32_t count1, int32_t count2, float dt) {
  // Encoder-derived base velocity. Wheel speed is measured relative to the
  // body, so the gyro pitch rate is added to get the wheel's rotation in the
  // world frame: axle ground speed = R * (joint_vel + pitch_rate), and with the
  // wheel sign convention that is exactly this expression. Without the
  // pitch_rate term, corr(vel_est, true velocity) was -0.58; with it, +0.74.
  lin_vel_ms_ = (-0.5f * (w1 + w2) + pitch_rate) * WHEEL_RADIUS;

  // obs[7]: distance from the arming point, from the encoder counts. It is the
  // exact integral of lin_vel_ms_ above (wheel rotation relative to the body
  // plus the body's own rotation), but taken from positions, so gyro bias and
  // velocity quantisation never accumulate in it. Signs as for w1/w2.
  if (!ref_latched_) {
    count1_ref_ = count1;
    count2_ref_ = count2;
    pitch_ref_  = pitch;
    ref_latched_ = true;
  }
  const float wheel_angle =
      0.5f * (ENC1_SIGN * (float)(count1 - count1_ref_) +
              ENC2_SIGN * (float)(count2 - count2_ref_)) *
      (2.0f * PI / ENCODER_COUNTS_PER_REV);
  x_disp_m_ = WHEEL_RADIUS * (-wheel_angle + (pitch - pitch_ref_));

  // obs[9]: integral of x_disp, clamped (anti-windup). Uses the un-normalised,
  // un-clamped x_disp in metres, as training does.
  x_int_ms_ += x_disp_m_ * dt;
  if (x_int_ms_ >  XINT_LIMIT) x_int_ms_ =  XINT_LIMIT;
  if (x_int_ms_ < -XINT_LIMIT) x_int_ms_ = -XINT_LIMIT;

  obs_[0] = c(pitch,       PITCH_LIMIT);   // rad    / 0.3
  obs_[1] = c(pitch_rate,  RATE_LIMIT);    // rad/s  / 10.0
  obs_[2] = c(roll,        PITCH_LIMIT);   // rad    / 0.3
  obs_[3] = c(roll_rate,   RATE_LIMIT);    // rad/s  / 10.0
  obs_[4] = c(w1,          VEL_LIMIT);     // rad/s  / 5.7
  obs_[5] = c(w2,          VEL_LIMIT);     // rad/s  / 5.7
  obs_[6] = c(lin_vel_ms_, LIN_VEL_LIM);   // m/s    / 0.19
  obs_[7] = c(x_disp_m_,   DRIFT_LIMIT);   // m      / 0.3
  // The command on the wheels NOW — the delayed one, not the one about to be
  // computed. This is what makes the observation Markov under ACTION_DELAY.
  obs_[8] = c(applied_cmd_, MAX_VEL);      // rad/s  / 5.7
  obs_[9] = c(x_int_ms_,   XINT_LIMIT);    // m*s    / 0.5
}

void Policy::zeroWheelObs() {
  for (int i = 4; i < OBS_DIM_CFG; i++) obs_[i] = 0.0f;
}

bool Policy::inferCommand(float* cmd_rad_s) {
  for (int i = 0; i < OBS_DIM_CFG; i++) g_input->data.f[i] = obs_[i];

  const uint32_t t0 = micros();
  g_op_profiler.startInference();
  const TfLiteStatus status = g_interpreter->Invoke();
  g_op_profiler.endInference();
  last_infer_us_ = micros() - t0;

  // First call after boot: the real measured latency, once. Gated: the control
  // loop cannot afford a Serial.printf on the very call it is timing.
  static bool printed_first = false;
  if (!printed_first) {
    printed_first = true;
    DBG_PRINTF("Policy: first inference took %lu us (budget %lu us) -> %s\n",
               (unsigned long)last_infer_us_, (unsigned long)CONTROL_DT_US,
               (last_infer_us_ < CONTROL_DT_US) ? "FITS" : "TOO SLOW");
  }
  // Flag later overruns too, but rate-limited: printing on every over-budget
  // call would turn one slow inference into a sustained overrun.
  static uint32_t slow_count = 0;
  if (last_infer_us_ > CONTROL_DT_US) {
    slow_count++;
    if ((slow_count % 200) == 1) {
      DBG_PRINTF("Policy: SLOW inference %lu us > %lu us budget (x%lu)\n",
                 (unsigned long)last_infer_us_, (unsigned long)CONTROL_DT_US,
                 (unsigned long)slow_count);
    }
  }

  if (status != kTfLiteOk) {
    Serial.println("Policy: Invoke() returned an error status");
    return false;
  }

  // The graph already clips to [-1, 1] (its int8 extremes are +/-0.9961, so
  // full command is 5.678 rad/s). Clamped again here as a guard, so the
  // command can never leave +/-MAX_VEL whatever model is linked.
  float y = g_output->data.f[0];
  if (y > 1.0f) y = 1.0f;
  if (y < -1.0f) y = -1.0f;
  raw_output_ = y;
  *cmd_rad_s = y * MAX_VEL;
  return true;
}

float Policy::applyDelay(float cmd_rad_s) {
  // Read the oldest entry, shift, append: a buffer of exactly ACTION_DELAY
  // entries delays by ACTION_DELAY ticks, matching training's
  //     q = [0] * ACTION_DELAY; q.append(cmd); applied = q.pop(0)
  // The command taking effect now was computed two ticks (10 ms) ago.
  const float applied = delay_buf_[0];
  for (int i = 0; i + 1 < ACTION_DELAY; i++) delay_buf_[i] = delay_buf_[i + 1];
  delay_buf_[ACTION_DELAY - 1] = cmd_rad_s;

  applied_cmd_ = applied;
  return applied;
}
