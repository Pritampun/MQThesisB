#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

RF24 radio(49,48);  // CE, CSN
const byte address[6] = "00001";

char inputBuffer[32];       // To hold input from Serial
char lastSent[32] = "";     // Keep track of last sent command

void setup() {
  Serial.begin(115200);
  delay(200);  // Let radio and serial stabilize

  Serial.println("Transmitter starting...");

  if (!radio.begin()) {
    Serial.println("ERROR: Radio hardware not responding!");
    while (1); // Halt if radio not found
  }

  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.setRetries(5, 15);
  radio.openWritingPipe(address);
  radio.stopListening();

  Serial.println("Radio initialized.");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); // Read until newline 
    input.trim();  // Remove any trailing newline or spaces

    if (input.length() > 0 && input != String(lastSent)) {
      input.toCharArray(inputBuffer, sizeof(inputBuffer));
      radio.write(&inputBuffer, strlen(inputBuffer)+1);
    }
    Serial.println(inputBuffer);
  }
}


