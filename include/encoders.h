// encoders.h — quadrature wheel encoders, counts and filtered velocity.
#pragma once

#include <Arduino.h>

class Encoders {
 public:
  void begin();

  // Call once per control tick. Converts the count delta since the last call
  // into rad/s at the wheel and applies the EMA.
  void update(float dt);

  float vel1() const { return vel1_; }   // rad/s, signed, forward positive
  float vel2() const { return vel2_; }

  int32_t count1() const;                // raw counts, for CPR verification
  int32_t count2() const;

  void resetCounts();

 private:
  int32_t last1_ = 0, last2_ = 0;
  float vel1_ = 0.0f, vel2_ = 0.0f;
};
