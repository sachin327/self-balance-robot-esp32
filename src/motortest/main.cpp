#include <Arduino.h>
#include <ESP32Encoder.h>

const int PWMA = 25, AIN1 = 26, AIN2 = 27;
const int PWMB = 33, BIN1 = 32, BIN2 = 4;
const int STBY = 13;
const int ENC_L_A = 18, ENC_L_B = 19;
const int ENC_R_A = 16, ENC_R_B = 17;

const int PWM_FREQ = 20000, PWM_RES = 10;
const int CH_A = 0, CH_B = 1;

ESP32Encoder encL, encR;

void motorA(int duty) {
  bool fwd = duty >= 0;
  digitalWrite(AIN1, fwd);
  digitalWrite(AIN2, !fwd);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PWMA, abs(duty));
#else
  ledcWrite(CH_A, abs(duty));
#endif
}

void motorB(int duty) {
  bool fwd = duty >= 0;
  digitalWrite(BIN1, fwd);
  digitalWrite(BIN2, !fwd);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PWMB, abs(duty));
#else
  ledcWrite(CH_B, abs(duty));
#endif
}

void runTest(const char *label, void (*motor)(int), ESP32Encoder &enc, int duty) {
  enc.clearCount();
  long before = enc.getCount();
  unsigned long t0 = millis();

  motor(duty);
  delay(2000);
  motor(0);
  delay(600);                      // let it coast to a stop

  long after = enc.getCount();
  float secs = (millis() - t0) / 1000.0;
  long delta = after - before;
  float revs = delta / 4250.0;     // counts per wheel revolution
  float rpm  = (revs / secs) * 60.0;

  Serial.printf("%-12s duty %5d | counts %8ld | %6.2f rev | %6.1f rpm\n",
                label, duty, delta, revs, rpm);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  for (int p : {AIN1, AIN2, BIN1, BIN2, STBY}) pinMode(p, OUTPUT);
  digitalWrite(STBY, LOW);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PWMA, PWM_FREQ, PWM_RES);
  ledcAttach(PWMB, PWM_FREQ, PWM_RES);
#else
  ledcSetup(CH_A, PWM_FREQ, PWM_RES);  ledcAttachPin(PWMA, CH_A);
  ledcSetup(CH_B, PWM_FREQ, PWM_RES);  ledcAttachPin(PWMB, CH_B);
#endif

  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  encL.attachFullQuad(ENC_L_A, ENC_L_B);
  encR.attachFullQuad(ENC_R_A, ENC_R_B);

  motorA(0); motorB(0);
  digitalWrite(STBY, HIGH);
  Serial.println("\nwheels off the ground\n");
  delay(1500);
}

void loop() {
  runTest("LEFT fwd",  motorA, encL,  400);   delay(800);
  runTest("LEFT rev",  motorA, encL, -400);   delay(800);
  runTest("RIGHT fwd", motorB, encR,  400);   delay(800);
  runTest("RIGHT rev", motorB, encR, -400);   delay(800);

  Serial.println("---- full duty ----");
  runTest("LEFT max",  motorA, encL, 1023);   delay(800);
  runTest("RIGHT max", motorB, encR, 1023);   delay(2000);
  Serial.println();
}