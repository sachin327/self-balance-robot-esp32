#include "motors.h"

#include <math.h>

#include "robot_config.h"

void Motors::begin() {
  pinMode(MOTOR_STBY_PIN, OUTPUT);
  pinMode(MOTOR_A_IN1_PIN, OUTPUT);
  pinMode(MOTOR_A_IN2_PIN, OUTPUT);
  pinMode(MOTOR_B_IN1_PIN, OUTPUT);
  pinMode(MOTOR_B_IN2_PIN, OUTPUT);

  // Arduino core 2.0.17: channel-based LEDC API.
  ledcSetup(MOTOR_A_LEDC_CH, MOTOR_PWM_FREQ_HZ, MOTOR_PWM_BITS);
  ledcSetup(MOTOR_B_LEDC_CH, MOTOR_PWM_FREQ_HZ, MOTOR_PWM_BITS);
  ledcAttachPin(MOTOR_A_PWM_PIN, MOTOR_A_LEDC_CH);
  ledcAttachPin(MOTOR_B_PWM_PIN, MOTOR_B_LEDC_CH);

  digitalWrite(MOTOR_STBY_PIN, HIGH);   // take TB6612 out of standby
  stop();
}

void Motors::driveChannel(int ledc_ch, int in1_pin, int in2_pin,
                          float sign, int duty) {
  if (sign < 0.0f) duty = -duty;

  int mag = abs(duty);
  if (mag > MOTOR_PWM_MAX) mag = MOTOR_PWM_MAX;

  if (mag == 0) {
    // Both inputs low = coast. Braking here would fight the policy, which was
    // trained against a velocity drive that simply stops applying torque.
    digitalWrite(in1_pin, LOW);
    digitalWrite(in2_pin, LOW);
    ledcWrite(ledc_ch, 0);
    return;
  }

  if (duty > 0) {
    digitalWrite(in1_pin, HIGH);
    digitalWrite(in2_pin, LOW);
  } else {
    digitalWrite(in1_pin, LOW);
    digitalWrite(in2_pin, HIGH);
  }
  ledcWrite(ledc_ch, mag);
}

void Motors::setVelocity(float target1, float target2,
                         float measured1, float measured2, float dt) {
  const float err1 = target1 - measured1;
  const float err2 = target2 - measured2;

  integ1_ += err1 * dt;
  integ2_ += err2 * dt;

  // Anti-windup: without this, a wheel held still by a hand or a stall winds
  // the integrator up and the robot lurches the moment it comes free. The clamp
  // is on the integral's PWM contribution, not on the raw integral, so it stays
  // meaningful if VEL_KI is retuned.
  const float i_max = VEL_I_CLAMP_PWM / VEL_KI;
  integ1_ = fmaxf(-i_max, fminf(i_max, integ1_));
  integ2_ = fmaxf(-i_max, fminf(i_max, integ2_));

  float u1 = VEL_KFF * target1 + VEL_KP * err1 + VEL_KI * integ1_;
  float u2 = VEL_KFF * target2 + VEL_KP * err2 + VEL_KI * integ2_;

  // Below the deadband the motor produces no motion at all, so step over it
  // rather than emitting duties that just make the driver warm.
  auto applyDeadband = [](float u) -> int {
    if (fabsf(u) < 1.0f) return 0;
    const int mag = MOTOR_DEADBAND + (int)(fabsf(u));
    return (u > 0.0f) ? mag : -mag;
  };

  duty1_ = applyDeadband(u1);
  duty2_ = applyDeadband(u2);

  if (duty1_ > MOTOR_PWM_MAX)  duty1_ = MOTOR_PWM_MAX;
  if (duty1_ < -MOTOR_PWM_MAX) duty1_ = -MOTOR_PWM_MAX;
  if (duty2_ > MOTOR_PWM_MAX)  duty2_ = MOTOR_PWM_MAX;
  if (duty2_ < -MOTOR_PWM_MAX) duty2_ = -MOTOR_PWM_MAX;

  driveChannel(MOTOR_A_LEDC_CH, MOTOR_A_IN1_PIN, MOTOR_A_IN2_PIN,
               MOTOR_A_SIGN, duty1_);
  driveChannel(MOTOR_B_LEDC_CH, MOTOR_B_IN1_PIN, MOTOR_B_IN2_PIN,
               MOTOR_B_SIGN, duty2_);
}

void Motors::setRawDuty(int d1, int d2) {
  duty1_ = d1;
  duty2_ = d2;
  driveChannel(MOTOR_A_LEDC_CH, MOTOR_A_IN1_PIN, MOTOR_A_IN2_PIN, MOTOR_A_SIGN, d1);
  driveChannel(MOTOR_B_LEDC_CH, MOTOR_B_IN1_PIN, MOTOR_B_IN2_PIN, MOTOR_B_SIGN, d2);
}

void Motors::stop() {
  integ1_ = 0.0f;
  integ2_ = 0.0f;
  duty1_ = 0;
  duty2_ = 0;
  driveChannel(MOTOR_A_LEDC_CH, MOTOR_A_IN1_PIN, MOTOR_A_IN2_PIN, MOTOR_A_SIGN, 0);
  driveChannel(MOTOR_B_LEDC_CH, MOTOR_B_IN1_PIN, MOTOR_B_IN2_PIN, MOTOR_B_SIGN, 0);
}
