// policy.h — observation assembly, TFLite Micro inference, and the command
// delay line. This is the translation layer between real sensors and the 10
// numbers the network was trained on.
#pragma once

#include <Arduino.h>

#include "robot_config.h"

class Policy {
 public:
  bool begin();

  // Assemble the 10-D observation. Every element is divided by its limit and
  // clamped to [-1, 1], in the exact order the network expects. count1/count2
  // are the raw encoder counts; obs[7] (distance from the arming point) and
  // obs[9] (its integral) are derived from them. The first call after reset()
  // latches the reference, so x_disp starts at 0 wherever the counts are.
  void buildObs(float pitch, float pitch_rate,
                float roll, float roll_rate,
                float w1, float w2,
                int32_t count1, int32_t count2, float dt);

  // Overwrite the observation directly, bypassing buildObs. Used by
  // MODE_VALIDATE to feed the fixed test fixture, which must reach the network
  // exactly as generated rather than being re-derived from sensors.
  void setObs(const float* obs);

  // Tilt demo only: zero every input that comes from the wheels (wheel speeds,
  // velocity estimate, distance, applied command, distance integral), leaving
  // pitch, roll and their rates. With the wheels in the air those inputs run
  // away and swamp the tilt; without them the output follows tilt alone.
  void zeroWheelObs();

  // Runs the network on the current observation. On success writes the wheel
  // command in rad/s (network output clamped to [-1, 1], times MAX_VEL) to
  // *cmd_rad_s and returns true. Returns false if Invoke() failed.
  bool inferCommand(float* cmd_rad_s);

  // Pushes `cmd` into the delay line and returns the command to apply now:
  // the one computed ACTION_DELAY ticks ago. Also latches it as the "applied"
  // value that obs[8] reports next tick.
  float applyDelay(float cmd_rad_s);

  void reset();   // clears position, its integral, delay line and applied command

  const float* obs() const { return obs_; }
  float rawOutput() const { return raw_output_; }   // network output, [-1, 1]
  float xDisp() const { return x_disp_m_; }
  float xInt() const { return x_int_ms_; }
  float linVel() const { return lin_vel_ms_; }
  float appliedCmd() const { return applied_cmd_; }
  uint32_t lastInferenceUs() const { return last_infer_us_; }
  size_t arenaUsedBytes() const;

  // Per-operator timing. Off by default so it adds nothing to normal
  // inference; enable it, run some inferences, then print the breakdown.
  void setOpProfiling(bool on);
  void printOpProfile(int inferences) const;

 private:
  float obs_[OBS_DIM_CFG] = {0};
  float raw_output_ = 0.0f;

  float x_disp_m_   = 0.0f;   // distance from the arming point, metres (encoder counts)
  float x_int_ms_   = 0.0f;   // integral of x_disp, m*s, clamped to +/-XINT_LIMIT
  // Reference for x_disp, latched on the first buildObs() after reset().
  bool    ref_latched_ = false;
  int32_t count1_ref_  = 0;
  int32_t count2_ref_  = 0;
  float   pitch_ref_   = 0.0f;
  float lin_vel_ms_ = 0.0f;   // encoder+gyro linear velocity estimate, m/s

  // Exactly ACTION_DELAY entries: read the oldest, shift, append. With one
  // entry more than this the delay is ACTION_DELAY + 1 ticks.
  float delay_buf_[ACTION_DELAY] = {0};
  float applied_cmd_ = 0.0f;  // rad/s, the command on the wheels now

  uint32_t last_infer_us_ = 0;
};
