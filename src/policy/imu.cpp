#include "imu.h"

#include <Wire.h>
#include <math.h>

#include "robot_config.h"

namespace {

constexpr uint8_t REG_SMPLRT_DIV  = 0x19;
constexpr uint8_t REG_CONFIG      = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG= 0x1C;
constexpr uint8_t REG_ACCEL_XOUT  = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1  = 0x6B;
constexpr uint8_t REG_WHO_AM_I    = 0x75;

constexpr float DEG_TO_RAD_F = 0.017453292519943295f;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

}  // namespace

bool Imu::begin() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ);

  const uint8_t who = readReg(REG_WHO_AM_I);
  if (who != 0x68 && who != 0x69 && who != 0x70 && who != 0x71) {
    Serial.printf("IMU: WHO_AM_I = 0x%02X at address 0x%02X — no MPU responding\n",
                  who, MPU_ADDR);

    // Scan the bus rather than leaving "it didn't work" as the whole diagnosis.
    // The three usual causes look different here: nothing at all means wiring or
    // power, a device at 0x69 means AD0 is pulled high, and anything else means
    // the wrong chip is on these pins.
    Serial.printf("IMU: scanning I2C on SDA=%d SCL=%d ...\n",
                  I2C_SDA_PIN, I2C_SCL_PIN);
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        Serial.printf("IMU:   device at 0x%02X\n", addr);
        found++;
      }
    }
    if (found == 0) {
      Serial.println("IMU:   nothing on the bus — check VCC/GND, SDA->21, SCL->22,");
      Serial.println("IMU:   and that the module has pull-ups (most breakouts do).");
    } else {
      Serial.printf("IMU:   %d device(s) found. If one is 0x69, set MPU_ADDR to 0x69\n",
                    found);
      Serial.println("IMU:   in robot_config.h (AD0 tied high).");
    }
    return false;
  }

  writeReg(REG_PWR_MGMT_1, 0x00);   // wake
  delay(50);
  writeReg(REG_PWR_MGMT_1, 0x01);   // clock from gyro X — more stable than internal osc
  delay(10);

  // DLPF 44 Hz accel / 42 Hz gyro. The default (260 Hz) passes motor and
  // gearbox vibration straight into the tilt estimate; at a 200 Hz control rate
  // there is nothing useful above ~50 Hz anyway.
  writeReg(REG_CONFIG, 0x03);

  // With DLPF on, the gyro output rate is 1 kHz; divider 4 gives 200 Hz, one
  // fresh sample per control tick.
  writeReg(REG_SMPLRT_DIV, 0x04);

  writeReg(REG_GYRO_CONFIG,  GYRO_FS_SEL << 3);
  writeReg(REG_ACCEL_CONFIG, ACCEL_FS_SEL << 3);
  delay(50);

  Serial.printf("IMU: MPU ok (WHO_AM_I 0x%02X), gyro +/-%d dps, accel +/-%dg\n",
                who, 250 << GYRO_FS_SEL, 2 << ACCEL_FS_SEL);
  return true;
}

bool Imu::calibrateGyro() {
  constexpr int kSettle = 40;    // discarded: lets the DLPF and timing settle
  constexpr int kSamples = 400;  // 2 s at the control rate

  double sx = 0.0, sy = 0.0, sz = 0.0;
  float lo[3] = { 1e9f,  1e9f,  1e9f};
  float hi[3] = {-1e9f, -1e9f, -1e9f};
  float min_trust = 1.0f;
  int taken = 0;
  ImuReading r;

  // Same pattern as the control loop: spin until the tick, then one read.
  uint32_t next = micros();
  for (int i = 0; i < kSettle + kSamples; i++) {
    while ((int32_t)(micros() - next) < 0) {}
    next += CONTROL_DT_US;
    if (!read(r) || i < kSettle) continue;
    // read() already subtracts the current bias, so this accumulates the
    // residual and a re-run refines the previous estimate.
    const float w[3] = {r.wx, r.wy, r.wz};
    for (int k = 0; k < 3; k++) {
      if (w[k] < lo[k]) lo[k] = w[k];
      if (w[k] > hi[k]) hi[k] = w[k];
    }
    const float trust = accelTrust(r.ax, r.ay, r.az);
    if (trust < min_trust) min_trust = trust;
    sx += r.wx; sy += r.wy; sz += r.wz;
    taken++;
  }
  if (taken < kSamples / 2) {
    Serial.println("IMU: gyro calibration could not read the sensor");
    return false;
  }
  const float range = fmaxf(hi[0] - lo[0], fmaxf(hi[1] - lo[1], hi[2] - lo[2]));
  if (range > GYRO_CAL_MAX_RANGE || min_trust < FILTER_SEED_MIN_TRUST) {
    Serial.printf("IMU: robot moved during calibration (gyro range %.3f rad/s, "
                  "accel trust %.2f) — not used\n", range, min_trust);
    return false;
  }

  bias_wx_ += (float)(sx / taken);
  bias_wy_ += (float)(sy / taken);
  bias_wz_ += (float)(sz / taken);

  Serial.printf("IMU: gyro bias %.5f %.5f %.5f rad/s (%d samples at %d Hz, I2C %lu Hz)\n",
                bias_wx_, bias_wy_, bias_wz_, taken, (int)(1000000UL / CONTROL_DT_US),
                (unsigned long)I2C_FREQ_HZ);
  return true;
}

bool Imu::read(ImuReading &out) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;

  int16_t raw[7];
  for (int i = 0; i < 7; i++) {
    raw[i] = (int16_t)((Wire.read() << 8) | Wire.read());
  }
  // raw[3] is temperature — read to advance the burst, not used.

  const float g_to_ms2 = GRAVITY_MAG / ACCEL_LSB_PER_G;
  out.ax = raw[0] * g_to_ms2;
  out.ay = raw[1] * g_to_ms2;
  out.az = raw[2] * g_to_ms2;

  const float lsb_to_rads = DEG_TO_RAD_F / GYRO_LSB_PER_DPS;
  out.wx = raw[4] * lsb_to_rads - bias_wx_;
  out.wy = raw[5] * lsb_to_rads - bias_wy_;
  out.wz = raw[6] * lsb_to_rads - bias_wz_;
  return true;
}

float Imu::accelTrust(float ax, float ay, float az) {
  const float mag = sqrtf(ax * ax + ay * ay + az * az);
  const float mag_error = fabsf(mag - GRAVITY_MAG);
  const float span = ACCEL_TRUST_ZERO_BAND - ACCEL_TRUST_FULL_BAND;
  const float t = 1.0f - (mag_error - ACCEL_TRUST_FULL_BAND) / span;
  return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

bool Imu::trySeed(const ImuReading &r, float min_trust) {
  if (accelTrust(r.ax, r.ay, r.az) < min_trust) return false;
  pitch_ = atan2f(-r.ax, r.az);   // same axis convention as update()
  roll_  = atan2f( r.ay, r.az);
  pitch_rate_ = r.wy;             // so the first trapezoid step is consistent
  roll_rate_  = r.wx;
  return true;
}

void Imu::update(const ImuReading &r, float dt, bool init) {
  // Axis convention from imu_filter.py: pitch about Y, roll about X.
  const float gyro_roll_rate  = r.wx;
  const float gyro_pitch_rate = r.wy;

  const float pitch_acc = atan2f(-r.ax, r.az);
  const float roll_acc  = atan2f( r.ay, r.az);
  const float trust     = accelTrust(r.ax, r.ay, r.az);

  if (init) {
    pitch_ = trust * pitch_acc;
    roll_  = trust * roll_acc;
  } else {
    // Gyro integrated with the trapezoid rule between samples, then nudged
    // toward the accelerometer angle in proportion to how much the accel is
    // currently worth trusting.
    const float correction = (1.0f - FILTER_ALPHA) * trust;
    pitch_ = (1.0f - correction) *
                 (pitch_ + 0.5f * (gyro_pitch_rate + pitch_rate_) * dt) +
             correction * pitch_acc;
    roll_  = (1.0f - correction) *
                 (roll_ + 0.5f * (gyro_roll_rate + roll_rate_) * dt) +
             correction * roll_acc;
  }

  pitch_rate_ = gyro_pitch_rate;
  roll_rate_  = gyro_roll_rate;
}
