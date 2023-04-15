#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include <SimpleTimer.h>
#include <dht.h>

// You should get Auth Token in the Blynk App.
// Go to the Project Settings (nut icon).
char auth[] = "k3ru6AD4whWrIuQb-_1a2ydAHxFp-eqd";

// Your WiFi credentials.
// Set password to "" for open networks.
char ssid[] = "jeraths";
char pass[] = "aditya2k";

#define DHTPIN D2          // What digital pin we're connected to
dht DHT;
// Uncomment whatever type you're using!
//#define DHTTYPE DHT11     // DHT 11
//#define DHTTYPE DHT22   // DHT 22, AM2302, AM2321
//#define DHTTYPE DHT21   // DHT 21, AM2301
int alarmPin = 4;
int led1 = 5;
int led2 = 13;

SimpleTimer timer;

void sendSensor(){
  int chk= DHT.read11(D2);
  float h = DHT.humidity;
  float t = DHT.temperature; // or dht.readTemperature(true) for Fahrenheit

  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  Serial.println(t);
  Blynk.virtualWrite(V5, h);
  Blynk.virtualWrite(V6, t);

  // SETUP the ALARM Trigger and Send EMAIL 
  // and PUSH Notification

  if(t > 25){
    Blynk.email("jerathhimani29@gmail.com", "ESP8266 Alert", "Temperature over 28C!");
    //Blynk.notify("ESP8266 Alert - Temperature over 28C!");
  }
}

void setup(){
  Serial.begin(115200);
  Blynk.begin(auth, ssid, pass);
//  DHT.begin();
  timer.setInterval(2500L, sendSensor);
}

void loop(){
  Blynk.run();
timer.run();
}

