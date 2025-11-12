#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

// Motor control pins
// const int m11 = 22;//22
// const int m12 = 24;//24
// const int m21 = 26;
// const int m22 = 28;
// const int m31 = 30;
// const int m32 = 32; //33 for one with marker 32 for other
// const int m41 = 34;
// const int m42 = 36;
const int m11 = 45;
const int m12 = 47;
const int m21 = 38;
const int m22 = 40;
const int m31 = 42;
const int m32 = 44; //33 for one with marker 32 for other
const int m41 = 46;
const int m42 = 48;

// const int en1 = 6;
// const int en2 = 5;
// const int en3 = 3;
// const int en4 = 2;

const int en1 = 43;
const int en2 = 41;
const int en3 = 39;
const int en4 = 37;
// const int motorSpeed = 150;

RF24 radio(35, 36);  // CE, CSN
const byte address[6] = "00001";
char buffer[32];
char lastCommand[32] = "";

unsigned long lastReceivedTime = 0;
const unsigned long signalTimeout = 250;  // Time to stop if no command

void setup() {
  Serial.begin(115200);

  pinMode(m11, OUTPUT); pinMode(m12, OUTPUT);
  pinMode(m21, OUTPUT); pinMode(m22, OUTPUT);
  pinMode(m31, OUTPUT); pinMode(m32, OUTPUT);
  pinMode(m41, OUTPUT); pinMode(m42, OUTPUT);

  pinMode(en1, OUTPUT); pinMode(en2, OUTPUT);
  pinMode(en3, OUTPUT); pinMode(en4, OUTPUT);

  if (!radio.begin()) {
    Serial.println("NRF24L01 not responding!");
    while (1);
  }

  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.openReadingPipe(0, address);
  radio.startListening();

  Serial.println("Receiver ready.");
}

void loop() {
  // Check for new radio data
  if (radio.available()) {
    radio.read(&buffer, sizeof(buffer));

    Serial.print("Received: ");
    Serial.println(buffer);

    lastReceivedTime = millis();

    if (strcmp(buffer, lastCommand) != 0 || strcmp(lastCommand, "0000") == 0) {
      handleCommand(buffer);
      strcpy(lastCommand, buffer);
    }
  }

  // If signal not received recently, stop motors
  if (millis() - lastReceivedTime > signalTimeout && strcmp(lastCommand, "0000") != 0) {
    stopMotors();
    strcpy(lastCommand, "0000");
  }
}
int speed = 130; // 80 for others 100 for one with marker
int Rspeed= speed+20;
void handleCommand(const char* cmd) {
  if      (strncmp(cmd, "0011", 4) == 0) moveMotors(-Rspeed,-Rspeed,-Rspeed,-Rspeed);//LEFT
  else if (strncmp(cmd, "0100", 4) == 0) moveMotors(Rspeed,Rspeed,Rspeed,Rspeed);//RIGHT
  else if (strncmp(cmd, "0010", 4) == 0) moveMotors(-speed,speed,-speed,speed);//BACK
  else if (strncmp(cmd, "0001", 4) == 0) moveMotors(speed,-speed,speed,-speed);//FORWARD
  // else if (strncmp(cmd, "0101", 4) == 0) right();
  // else if (strncmp(cmd, "0110", 4) == 0) left();
  else                                   stopMotors();  // Unknown or stop
}

void setMotor(int in1, int in2, int en, int pwm) {
  if (pwm > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(en, pwm);
  } else if (pwm < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(en, -pwm);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(en, 0);
  }
}

void moveMotors(int topLeftPWM, int topRightPWM, int bottomLeftPWM, int bottomRightPWM) {

  setMotor(m11, m12, en1, topLeftPWM);
  setMotor(m41, m42, en4, topRightPWM);
  setMotor(m21, m22, en2, bottomLeftPWM);
  setMotor(m31, m32, en3, bottomRightPWM);

}

void rotateClockwise(int speed) {
  moveMotors(speed, -speed, speed, -speed);
}

void rotateCounterClockwise(int speed) {
  moveMotors(-speed, speed, -speed, speed);
}

void stopMotors() {
  moveMotors(0, 0, 0, 0);
}