// imu.h — MPU6050 driver plus the complementary filter, ported from
// imu_filter.py so the policy sees the same orientation estimate it trained on.
#pragma once

#include <Arduino.h>

struct ImuReading {
  float ax, ay, az;   // m/s^2  — INCLUDING gravity
  float wx, wy, wz;   // rad/s  — gyro bias removed
};

class Imu {
 public:
  bool begin();

  // Measures and stores gyro bias. The robot must be still.
  //
  // Gyro bias only — the accelerometer is deliberately NOT zeroed. Tilt is
  // derived from the direction of gravity, so subtracting the at-rest
  // accelerometer reading would remove the very signal the filter needs and
  // leave atan2(-ax, az) computing an angle from noise.
  void calibrateGyro(int samples = 1000);

  bool read(ImuReading &out);

  // Advance the filter one control step from raw sensor values.
  // `init` does a one-shot accel-only attitude set, matching the reset path in
  // imu_filter.py, so the estimate does not spend seconds converging at boot.
  void update(const ImuReading &r, float dt, bool init = false);

  float pitch() const { return pitch_; }
  float roll() const { return roll_; }
  float pitchRate() const { return pitch_rate_; }
  float rollRate() const { return roll_rate_; }

  float gyroBiasX() const { return bias_wx_; }
  float gyroBiasY() const { return bias_wy_; }
  float gyroBiasZ() const { return bias_wz_; }

  // 1.0 = accelerometer is pure gravity, 0.0 = swamped by motion.
  static float accelTrust(float ax, float ay, float az);

 private:
  float pitch_ = 0.0f, roll_ = 0.0f;
  float pitch_rate_ = 0.0f, roll_rate_ = 0.0f;
  float bias_wx_ = 0.0f, bias_wy_ = 0.0f, bias_wz_ = 0.0f;
};
