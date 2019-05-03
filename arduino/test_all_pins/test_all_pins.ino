// Test all pins

int maxpins = 23;
void setup() {
  Serial.begin(9600);
  Serial.write("Setting up");
  for(int i = 0; i < maxpins; i++) {
    pinMode(i, OUTPUT);
    digitalWrite(i, HIGH);
  }
}

void loop() {
	if (Serial.available()){
		char recievedChar = Serial.read();
		Serial.print("Read: (");
		Serial.print(recievedChar);
		Serial.print(") ");
		if (recievedChar == 255) {
			turnAllOff();
			Serial.println("Turned all off");
		} else {
			int i = recievedChar;
			digitalWrite(i, LOW);
			Serial.print("Turned on pin ");
			Serial.println(i);
		}
	}
}

void turnAllOff()
{
  for(int i = 0; i < maxpins; i++) {
    digitalWrite(i, HIGH);
  }
}
