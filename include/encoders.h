// encoders.h — quadrature wheel encoders, counts and filtered velocity.
#pragma once

#include <Arduino.h>

#include "robot_config.h"

class Encoders {
 public:
  void begin();

  // Call once per control tick (5 ms). Velocity is measured over the last
  // ENC_WINDOW_TICKS ticks (20 ms), not over dt: at a 5 ms window one count is
  // 0.296 rad/s, coarser than two whole action levels. The result is then
  // quantised to VEL_QUANTUM to match what train.py fed the network.
  //
  // dt is the control period, used to derive the window duration. The wider
  // window is handled internally by a count ring, so the call site does not
  // change — keep calling this every tick.
  void update(float dt);

  // rad/s, signed, forward positive, quantised to multiples of VEL_QUANTUM.
  // These feed obs[4] and obs[5] directly, and obs[6] is derived from them, so
  // the quantisation propagates exactly as it does in train.py.
  float vel1() const { return vel1_; }
  float vel2() const { return vel2_; }

  int32_t count1() const;                // raw counts, for CPR verification
  int32_t count2() const;

  void resetCounts();

 private:
  // Ring of the last ENC_WINDOW_TICKS + 1 count samples. The oldest entry is
  // ENC_WINDOW_TICKS ticks back, so the delta against it spans the full window.
  int32_t ring1_[ENC_WINDOW_TICKS + 1] = {0};
  int32_t ring2_[ENC_WINDOW_TICKS + 1] = {0};
  int     ring_head_ = 0;    // index of the newest sample
  int     ring_fill_ = 0;    // samples collected; window valid once it saturates

  float vel1_ = 0.0f, vel2_ = 0.0f;
};
