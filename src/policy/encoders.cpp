#include "encoders.h"

#include <math.h>

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
  for (int i = 0; i <= ENC_WINDOW_TICKS; i++) {
    ring1_[i] = 0;
    ring2_[i] = 0;
  }
  ring_head_ = 0;
  ring_fill_ = 0;
  vel1_ = 0.0f;
  vel2_ = 0.0f;
}

void Encoders::update(float dt) {
  const int32_t c1 = count1();
  const int32_t c2 = count2();

  // Advance the ring, then read back the sample ENC_WINDOW_TICKS entries ago.
  // With ENC_WINDOW_TICKS + 1 slots the tail is exactly one window behind the
  // head once the ring has filled.
  ring_head_ = (ring_head_ + 1) % (ENC_WINDOW_TICKS + 1);
  ring1_[ring_head_] = c1;
  ring2_[ring_head_] = c2;
  if (ring_fill_ <= ENC_WINDOW_TICKS) ring_fill_++;

  // Until the ring fills — the first 4 ticks after boot or an arm — measure over
  // however many samples exist. Velocity is briefly noisier rather than wrong,
  // and it resolves within 20 ms.
  const int span = (ring_fill_ > ENC_WINDOW_TICKS) ? ENC_WINDOW_TICKS
                                                   : (ring_fill_ - 1);
  if (span <= 0) {
    vel1_ = 0.0f;
    vel2_ = 0.0f;
    return;
  }

  const int tail = (ring_head_ - span + (ENC_WINDOW_TICKS + 1)) %
                   (ENC_WINDOW_TICKS + 1);
  const int32_t d1 = c1 - ring1_[tail];
  const int32_t d2 = c2 - ring2_[tail];

  // counts -> revolutions -> rad/s, over the window rather than a single tick
  const float window_s = dt * (float)span;
  const float scale = (2.0f * PI) / (ENCODER_COUNTS_PER_REV * window_s);
  float w1 = ENC1_SIGN * d1 * scale;
  float w2 = ENC2_SIGN * d2 * scale;

  // Quantise last, exactly as train.py:768-770 does, and unconditionally: the
  // staircase is hardware, not noise. obs[6]'s velocity estimate is derived from
  // these in Policy::buildObs(), so it inherits the same steps the simulator had.
  w1 = roundf(w1 / VEL_QUANTUM) * VEL_QUANTUM;
  w2 = roundf(w2 / VEL_QUANTUM) * VEL_QUANTUM;

  vel1_ = w1;
  vel2_ = w2;
}
