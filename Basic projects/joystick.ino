#include <Servo.h> // include servo library

Servo servo1; // define servos
//Servo servo2;

int joyX = 0; // give variable to joystick readings
int joyY = 1;

int joyVal;
int joyVal1;
// create variable for joystick value

void setup()
{
servo1.attach(3); // start servos
//servo2.attach(4);



}


void loop()
{
  
joyVal = analogRead(joyX); // read value from joystick
joyVal = map(joyVal, 0, 1023, 0, 180); // change value range to 0-180
servo1.write(joyVal); // write value to servo

joyVal1 = analogRead(joyY); // repeat same for y axis
joyVal1 = map(joyVal1, 0, 1023, 0, 180);
servo1.write(joyVal1);
delay(20); // add small delay to reduce noise



  
  
}
