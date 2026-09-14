// policy.h — observation assembly, TFLite Micro inference, and the action-delay
// buffer. This is the translation layer between real sensors and the 9 numbers
// the network was trained on.
#pragma once

#include <Arduino.h>

#include "robot_config.h"

class Policy {
 public:
  bool begin();

  // Assemble the 9-D observation. Every element is divided by its limit from
  // train.py and clamped to [-1, 1], in the exact order the network expects.
  // Also advances the odometry that feeds obs[7].
  void buildObs(float pitch, float pitch_rate,
                float roll, float roll_rate,
                float w1, float w2, float dt);

  // Overwrite the observation directly, bypassing buildObs. Used by
  // MODE_VALIDATE to feed the fixed test fixture, which must reach the network
  // exactly as generated rather than being re-derived from sensors.
  void setObs(const float* obs);

  // Runs the network on the current observation and returns argmax over the 13
  // Q-values. Returns -1 if Invoke() failed.
  int inferActionIdx();

  // Pushes `chosen` into the delay line and returns the index that should be
  // applied now — the one chosen ACTION_DELAY steps ago. Also latches it as the
  // "last applied" value that obs[8] reports next tick.
  int applyDelay(int chosen);

  void reset();   // clears odometry, delay line and last-applied action

  const float* obs() const { return obs_; }
  const float* qValues() const { return q_; }
  float xDisp() const { return x_disp_m_; }
  float linVel() const { return lin_vel_ms_; }
  int lastAppliedIdx() const { return last_applied_idx_; }
  uint32_t lastInferenceUs() const { return last_infer_us_; }
  size_t arenaUsedBytes() const;

 private:
  float obs_[OBS_DIM_CFG] = {0};
  float q_[N_ACTIONS_CFG] = {0};

  float x_disp_m_   = 0.0f;   // integrated drift from start, metres
  float lin_vel_ms_ = 0.0f;   // encoder+gyro linear velocity estimate, m/s

  int delay_buf_[ACTION_DELAY + 1];
  int last_applied_idx_ = ACTION_ZERO_IDX;

  uint32_t last_infer_us_ = 0;
};
