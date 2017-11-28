// int rTrigger = 23;
// int lTrigger = 22;
// int rBumper = 21;
// int lBumper = 20;

// int xButton = 19;
// int yButton = 18;
// int aButton = 17;
// int bButton = 16;

// int right = 15;
// int left = 14;
// int down= 13;
// int up = 12;

// int backButton = 9;
// int startButton = 8;

// const int myPins[] = {
//   rTrigger,
//   lTrigger,
//   rBumper,
//   lBumper,
//   xButton,
//   yButton,
//   aButton,
//   bButton,
//   right,
//   left,
//   down,
//   up,
//   backButton,
//   startButton
// };


// const char* myPinNames[] = {
//   "rTrigger",
//   "lTrigger",
//   "rBumper",
//   "lBumper",
//   "xButton",
//   "yButton",
//   "aButton",
//   "bButton",
//   "right",
//   "left",
//   "down",
//   "up",
//   "backButton",
//   "startButton"
// };

// int button_length = 12;

// void setup() {
  
//   Serial.begin(9600);
//   Serial.write("Setting up");
//   for(int i = 0; i < button_length; i++) {
//     pinMode(myPins[i], OUTPUT);
//     digitalWrite(myPins[i], HIGH);
//   }
// }

// void loop() {
//   if (Serial.available() > 0) {
//     // read the incoming byte:
//     handleInput(Serial.read());
//   }
// }

// void handleInput(byte b)
// {
//   Serial.print("Read: (");
//   Serial.print(b);
//   Serial.print(") ");

//     if (b == 0) {
//       turnAllOff();
//       Serial.println("All off");
//       return;
//     }

//   char bitstr[8];
//   b2s(b, bitstr);
//   // for(int i = 0; i<8; i++){
//   //  bitstr[i]='0';
//   //  if(b == (1<<i)){
//   //    bitstr[i]='1';
//   //  }
//   // }
//   // // terminate the string with the null character
//   // bitstr[7] = '\0';

//   Serial.print(bitstr);
//   Serial.print(" Buttons: ");
//   bool high = ((1 << 7) & b);

//   // low bit: buttons and bumpers
//   // high bit: arrows, triggers
//   for(int i = 0; i < 7; i++) {
//     int pinNum = i;
//     if (high)
//       pinNum += 7;

//     digitalWrite(myPins[pinNum], ((1 << i) & b) ? LOW : HIGH);
//     if ((1 << i) & b) {
//       Serial.print("(");
//       Serial.print(pinNum);
//       Serial.print("):");
//       Serial.print(myPinNames[pinNum]);     
//       Serial.print(" ");   
//     }
//   }
  
//   Serial.println(" Done");
// }

// void turnAllOff()
// {
//   for(int i = 0; i < button_length; i++) {
//     digitalWrite(myPins[i], HIGH);
//   }
// }

// void b2s(byte myByte, char* out){
//  byte mask = B10000000;
//  for(int i = 0; i<8; i++){
//    out[i]='0';
//    if(((mask >> i) & myByte) == (mask>>i)){
//      out[i]='1';
//    }
//  }
//  // terminate the string with the null character
//  out[8] = '\0';
// }

