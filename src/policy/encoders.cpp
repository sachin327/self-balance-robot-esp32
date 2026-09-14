#include "encoders.h"

#include "robot_config.h"

namespace {

// Full 4x quadrature decode: both edges of both channels. A 2x or 1x decode
// would throw away resolution that already gets thin at the slowest command
// level (~0.5 counts per tick at 0.21 rad/s).
volatile int32_t g_count1 = 0;
volatile int32_t g_count2 = 0;
volatile uint8_t g_state1 = 0;
volatile uint8_t g_state2 = 0;

// Indexed by (prev_state << 2) | new_state, where each state is (A << 1) | B.
// 0 marks the invalid double-transitions that only occur on a missed edge.
const int8_t kQuadTable[16] = {
   0, -1, +1,  0,
  +1,  0,  0, -1,
  -1,  0,  0, +1,
   0, +1, -1,  0,
};

void IRAM_ATTR isr1() {
  const uint8_t s = (uint8_t)((digitalRead(ENC1_A_PIN) << 1) | digitalRead(ENC1_B_PIN));
  g_count1 += kQuadTable[(g_state1 << 2) | s];
  g_state1 = s;
}

void IRAM_ATTR isr2() {
  const uint8_t s = (uint8_t)((digitalRead(ENC2_A_PIN) << 1) | digitalRead(ENC2_B_PIN));
  g_count2 += kQuadTable[(g_state2 << 2) | s];
  g_state2 = s;
}

int32_t readAtomic(volatile int32_t &v) {
  noInterrupts();
  const int32_t c = v;
  interrupts();
  return c;
}

}  // namespace

void Encoders::begin() {
  pinMode(ENC1_A_PIN, INPUT_PULLUP);
  pinMode(ENC1_B_PIN, INPUT_PULLUP);
  pinMode(ENC2_A_PIN, INPUT_PULLUP);
  pinMode(ENC2_B_PIN, INPUT_PULLUP);

  g_state1 = (uint8_t)((digitalRead(ENC1_A_PIN) << 1) | digitalRead(ENC1_B_PIN));
  g_state2 = (uint8_t)((digitalRead(ENC2_A_PIN) << 1) | digitalRead(ENC2_B_PIN));

  attachInterrupt(digitalPinToInterrupt(ENC1_A_PIN), isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1_B_PIN), isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_A_PIN), isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_B_PIN), isr2, CHANGE);

  resetCounts();
}

int32_t Encoders::count1() const { return readAtomic(g_count1); }
int32_t Encoders::count2() const { return readAtomic(g_count2); }

void Encoders::resetCounts() {
  noInterrupts();
  g_count1 = 0;
  g_count2 = 0;
  interrupts();
  last1_ = 0;
  last2_ = 0;
  vel1_ = 0.0f;
  vel2_ = 0.0f;
}

void Encoders::update(float dt) {
  const int32_t c1 = count1();
  const int32_t c2 = count2();

  const int32_t d1 = c1 - last1_;
  const int32_t d2 = c2 - last2_;
  last1_ = c1;
  last2_ = c2;

  // counts -> revolutions -> rad/s
  const float scale = (2.0f * PI) / (ENCODER_COUNTS_PER_REV * dt);
  const float raw1 = ENC1_SIGN * d1 * scale;
  const float raw2 = ENC2_SIGN * d2 * scale;

  vel1_ += ENC_VEL_EMA_ALPHA * (raw1 - vel1_);
  vel2_ += ENC_VEL_EMA_ALPHA * (raw2 - vel2_);
}
