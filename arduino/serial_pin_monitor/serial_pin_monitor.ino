

int backButton = 13;
int startButton = 12;

int yButton = 8;
int xButton = 10;
int aButton = 7;
int bButton = 11;

int upButton = 6;
int downButton = 5;
int leftButton = 3;
int rightButton = 4;

const int myPins[] = {
    yButton,
    xButton,
    bButton,
    aButton,
    leftButton,
    rightButton,
    upButton,
    downButton    
};


const char* buttonNames[] = {
    "y",
    "x",
    "b",
    "a",
    "left",
    "right",
    "up",
    "down"
};

int button_length = 8;

const byte startMask = B11000000;
const byte backMask = B00110000;

bool debug = 0;

void setup() {
    Serial.begin(9600);
//    Serial.write("Setting up");
    pinMode(upButton, OUTPUT);
    pinMode(downButton, OUTPUT);
    pinMode(leftButton, OUTPUT);
    pinMode(rightButton, OUTPUT);
    pinMode(aButton, OUTPUT);
    pinMode(bButton, OUTPUT);
    pinMode(xButton, OUTPUT);
    pinMode(yButton, OUTPUT);
    pinMode(backButton, OUTPUT);
    pinMode(startButton, OUTPUT);
    
    turnAllOff();
}

void loop() {
    if (Serial.available() > 0) {
        // read the incoming byte:
        handleInput(Serial.read());
    }
}

void handleInput(byte b)
{
//    b2s(b);

    //Serial.print("buttons: ");

    switch(b) {
        case 0:
            turnAllOff();
            //Serial.println("All off");
            break;
        case startMask:
            turnAllOff();
            digitalWrite(startButton, HIGH);
            //Serial.println("Start");
            break;
        case backMask:
            turnAllOff();
            digitalWrite(backButton, HIGH);
            //Serial.println("Back");
            break;
        default:  
            turnOn(b);
            break;
    } 
}

void turnOn(byte b)
{
    for(int i = 0; i < button_length; i++) {
       if ((1 << i) & b) {
            digitalWrite(myPins[i], HIGH);
            //Serial.print(buttonNames[i]);
            //Serial.print(" ");
       } else {
            digitalWrite(myPins[i], LOW);
       }
    }
    //Serial.println();
}

void turnAllOff()
{
    digitalWrite(startButton, LOW);
    digitalWrite(backButton, LOW);
    for(int i = 0; i < button_length; i++) {
       digitalWrite(myPins[i], LOW);
    }
}


void b2s(byte myByte){
 char Str[8];
 byte mask = B10000000;
 for(int i = 0; i<8; i++){
   Str[i]='0';
   if(((mask >> i) & myByte) == (mask>>i)){
     Str[i]='1';
   }
 }
 // terminate the string with the null character
 Str[8] = '\0';
 Serial.println(Str);
}
