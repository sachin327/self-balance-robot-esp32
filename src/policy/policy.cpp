#include "policy.h"

#include <math.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

// Measured on hardware: this graph reports arena_used_bytes = 7864.
constexpr int kArenaSize = 12 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];

// Eight operators — exactly what convert_tflite.py reports for this graph.
// LayerNorm has no TFLite op of its own; MEAN/SUB/MUL/ADD/SQRT/DIV are its
// decomposition. A resolver one entry short fails inside AllocateTensors().
constexpr int kNumOps = 8;

tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// train_obs9_baseline.py _c(): normalise then clamp to [-1, 1].
inline float c(float v, float limit) {
  const float x = v / limit;
  return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

}  // namespace

bool Policy::begin() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Policy: schema version %lu, expected %d\n",
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<kNumOps> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||   // the Linear layers
      resolver.AddRelu()           != kTfLiteOk ||   // activations
      resolver.AddMean()           != kTfLiteOk ||   // LayerNorm mean, advantage mean
      resolver.AddSub()            != kTfLiteOk ||   // x - mean
      resolver.AddMul()            != kTfLiteOk ||   // variance, * gamma
      resolver.AddAdd()            != kTfLiteOk ||   // + eps, + beta, V + A
      resolver.AddSqrt()           != kTfLiteOk ||   // sqrt(var + eps)
      resolver.AddDiv()            != kTfLiteOk) {   // normalise
    Serial.println("Policy: operator registration failed");
    return false;
  }

  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Policy: AllocateTensors() failed — arena too small, or the");
    Serial.println("        model uses an operator not registered above.");
    return false;
  }

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Catch a model/firmware mismatch here rather than as silent garbage later —
  // swapping in an 11-D checkpoint would otherwise read past the input tensor.
  if (g_input->type != kTfLiteFloat32 || g_output->type != kTfLiteFloat32) {
    Serial.println("Policy: expected float32 tensors");
    return false;
  }
  if (g_input->dims->data[1] != OBS_DIM_CFG) {
    Serial.printf("Policy: model wants %d inputs, firmware builds %d.\n",
                  g_input->dims->data[1], OBS_DIM_CFG);
    Serial.println("        (the 11-D variant appends pending[0], pending[1])");
    return false;
  }
  if (g_output->dims->data[1] != N_ACTIONS_CFG) {
    Serial.printf("Policy: model has %d actions, firmware expects %d\n",
                  g_output->dims->data[1], N_ACTIONS_CFG);
    return false;
  }

  reset();
  Serial.printf("Policy: ready — %u -> %u, arena %u/%d bytes\n",
                (unsigned)g_input->dims->data[1],
                (unsigned)g_output->dims->data[1],
                (unsigned)g_interpreter->arena_used_bytes(), kArenaSize);
  return true;
}

size_t Policy::arenaUsedBytes() const {
  return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}

void Policy::reset() {
  x_disp_m_ = 0.0f;
  lin_vel_ms_ = 0.0f;
  for (int i = 0; i <= ACTION_DELAY; i++) delay_buf_[i] = ACTION_ZERO_IDX;
  last_applied_idx_ = ACTION_ZERO_IDX;
  for (int i = 0; i < OBS_DIM_CFG; i++) obs_[i] = 0.0f;
}

void Policy::setObs(const float* obs) {
  for (int i = 0; i < OBS_DIM_CFG; i++) obs_[i] = obs[i];
}

void Policy::buildObs(float pitch, float pitch_rate,
                      float roll, float roll_rate,
                      float w1, float w2, float dt) {
  // Encoder-derived base velocity. Joint velocity is wheel-relative-to-body, so
  // the gyro pitch rate is added to get the wheel's rotation in the world frame.
  // The leading minus is the sign convention the training script verified:
  // without it corr(v_est, v_true) = -0.58 and velocity feedback destabilises;
  // with it, +0.74.
  lin_vel_ms_ = (-0.5f * (w1 + w2) + pitch_rate) * WHEEL_RADIUS;

  // Odometry for obs[7]. The simulator read true displacement from physics; on
  // hardware this integral is the only estimate available, and it drifts — so it
  // is re-zeroed on every arm.
  x_disp_m_ += lin_vel_ms_ * dt;

  obs_[0] = c(pitch,       PITCH_LIMIT);   // rad    / 1.2
  obs_[1] = c(pitch_rate,  RATE_LIMIT);    // rad/s  / 10.0
  obs_[2] = c(roll,        PITCH_LIMIT);   // rad    / 1.2
  obs_[3] = c(roll_rate,   RATE_LIMIT);    // rad/s  / 10.0
  obs_[4] = c(w1,          VEL_LIMIT);     // rad/s  / 5.7
  obs_[5] = c(w2,          VEL_LIMIT);     // rad/s  / 5.7
  obs_[6] = c(lin_vel_ms_, LIN_VEL_LIM);   // m/s    / 0.19
  obs_[7] = c(x_disp_m_,   DRIFT_LIMIT);   // m      / 2.0
  // The command driving the wheels NOW — post-delay, not the one just chosen.
  // This is what makes the observation Markov under ACTION_DELAY.
  obs_[8] = c(ACTIONS[last_applied_idx_], MAX_VEL);   // rad/s / 5.7
}

int Policy::inferActionIdx() {
  for (int i = 0; i < OBS_DIM_CFG; i++) g_input->data.f[i] = obs_[i];

  const uint32_t t0 = micros();
  const TfLiteStatus status = g_interpreter->Invoke();
  last_infer_us_ = micros() - t0;

  if (status != kTfLiteOk) return -1;

  // Q = V + (A - mean A). The value term is a large offset shared by all 13
  // entries and neighbouring actions differ by ~0.05 against returns of ~100,
  // so the ranking lives entirely in the advantage spread. Compare the raw
  // outputs; do not rescale them.
  int best = 0;
  for (int i = 0; i < N_ACTIONS_CFG; i++) {
    q_[i] = g_output->data.f[i];
    if (q_[i] > q_[best]) best = i;
  }
  return best;
}

int Policy::applyDelay(int chosen) {
  // ACTION_DELAY = 2: append the chosen action, apply the oldest. The command
  // taking effect now was decided two ticks (10 ms) ago.
  const int applied = delay_buf_[0];
  for (int i = 0; i < ACTION_DELAY; i++) delay_buf_[i] = delay_buf_[i + 1];
  delay_buf_[ACTION_DELAY] = chosen;

  last_applied_idx_ = applied;
  return applied;
}
