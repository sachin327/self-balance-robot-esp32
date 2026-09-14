// motors.h — TB6612FNG output with a per-wheel PI velocity controller.
//
// ACTIONS are wheel *velocity targets* in rad/s, because the simulator drove
// the joints with stiffness=0/damping=500 — a velocity servo. Writing a PWM
// duty proportional to the target instead would drift badly under load: the
// same duty gives very different wheel speeds on carpet versus a desk, and the
// policy would be commanding a speed it never actually gets.
#pragma once

#include <Arduino.h>

class Motors {
 public:
  void begin();

  // target: rad/s per wheel. measured: rad/s from the encoders.
  void setVelocity(float target1, float target2,
                   float measured1, float measured2, float dt);

  // Cut drive and clear the integrators. Used on fall, on watchdog, and in
  // MODE_DIAG so the sensor pipeline can be inspected with the wheels passive.
  void stop();

  int duty1() const { return duty1_; }
  int duty2() const { return duty2_; }

  // Raw open-loop drive, for deadband and direction checks in MODE_DIAG.
  void setRawDuty(int d1, int d2);

 private:
  void driveChannel(int ledc_ch, int in1_pin, int in2_pin, float sign, int duty);

  float integ1_ = 0.0f, integ2_ = 0.0f;
  int duty1_ = 0, duty2_ = 0;
};
