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
  //
  // Reads at the control loop's cadence (one read per CONTROL_DT_US after a
  // busy-wait), because this sensor's zero offset shifts with how it is read;
  // see the note at I2C_FREQ_HZ. Takes about 2 s. Returns false, leaving the
  // bias unchanged, if the robot moved during the window.
  bool calibrateGyro();

  // Use a previously measured bias (e.g. loaded from flash) instead of
  // calibrating at boot.
  void setGyroBias(float x, float y, float z) {
    bias_wx_ = x; bias_wy_ = y; bias_wz_ = z;
  }

  bool read(ImuReading &out);

  // Advance the filter one control step from raw sensor values.
  // `init` does a one-shot accel-only attitude set, matching the reset path in
  // imu_filter.py, so the estimate does not spend seconds converging at boot.
  void update(const ImuReading &r, float dt, bool init = false);

  // Seed the filter from the accelerometer alone, but only if the robot is
  // still enough to trust it (accel trust >= min_trust). Returns true when the
  // seed was accepted. Unlike update(..., init=true), the angle is NOT scaled
  // by trust: a trust-weighted seed shrinks the starting tilt toward zero when
  // the robot is being handled, and the 1 s arm delay (~one filter time
  // constant) leaves about a third of that error in place.
  bool trySeed(const ImuReading &r, float min_trust);

  // Chassis attitude: the filter's estimate minus the IMU mounting offset.
  // Everything downstream (observation, arming window, fall check) uses these.
  float pitch() const { return pitch_ - mount_pitch_ - balance_trim_; }
  float roll() const { return roll_ - mount_roll_; }

  // The filter's estimate of the IMU board's own attitude, offset NOT removed.
  // Used only to measure the mounting offset.
  float rawPitch() const { return pitch_; }
  float rawRoll() const { return roll_; }

  // How far the IMU board is rotated relative to the chassis, rad. Measured
  // with the chassis held truly vertical (see 'o' in policy_quick_check). The
  // filter itself keeps running on raw readings, so the accel/gyro maths are
  // untouched; only what is reported changes.
  void setMountOffset(float pitch_rad, float roll_rad) {
    mount_pitch_ = pitch_rad; mount_roll_ = roll_rad;
  }
  float mountPitch() const { return mount_pitch_; }
  float mountRoll() const { return mount_roll_; }

  // Balance trim, rad: the chassis tilt at which the robot actually balances,
  // i.e. where its centre of mass is over the axle. Zero for a symmetric
  // robot. A battery mounted off-centre moves it by degrees (the CoM is only
  // ~24 mm above the axle, so 1 mm of fore/aft offset is 2.4 deg). The policy
  // was trained with pitch = 0 at the balance point, so this is subtracted as
  // well. Pitch only; roll has no balance point.
  void setBalanceTrim(float pitch_rad) { balance_trim_ = pitch_rad; }
  float balanceTrim() const { return balance_trim_; }
  // Chassis pitch with the mount offset removed but not the balance trim.
  float chassisPitch() const { return pitch_ - mount_pitch_; }
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
  float mount_pitch_ = 0.0f, mount_roll_ = 0.0f;
  float balance_trim_ = 0.0f;
};
