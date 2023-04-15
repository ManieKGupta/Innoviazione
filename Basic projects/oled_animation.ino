#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels


// Declaration for SSD1306 display connected using software SPI 
#define OLED_MOSI   11
#define OLED_CLK   13
#define OLED_DC    9
#define OLED_CS    10
#define OLED_RESET 8
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);


// interval between the animation frames
int frame_delay =420;

const unsigned char Frame1 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0xff, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00, 
  0x00, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x01, 0x83, 0x80, 0x00, 0x00, 0x00, 
  0x07, 0x00, 0x00, 0x00, 0x03, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x01, 0xc0, 0x00, 0x00, 0x03, 0x0c, 
  0x40, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xe1, 0x0c, 0x60, 0x00, 0x00, 0x00, 0x00, 0xf0, 
  0x00, 0x1f, 0xff, 0x80, 0x20, 0x00, 0x00, 0x00, 0x01, 0xe0, 0x00, 0x01, 0xf9, 0xe0, 0x20, 0x00, 
  0x00, 0x00, 0x03, 0xc0, 0x00, 0x00, 0xcc, 0x78, 0x60, 0x00, 0x00, 0x03, 0xe7, 0x80, 0x00, 0x00, 
  0x64, 0x1f, 0xc0, 0x00, 0x00, 0x0e, 0xff, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x00, 0x1c, 
  0x0e, 0x00, 0x00, 0x00, 0x33, 0xf0, 0x00, 0x00, 0x00, 0x30, 0xf8, 0x00, 0x00, 0x00, 0x18, 0x1f, 
  0xf0, 0x00, 0x00, 0x7f, 0xf0, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0xfc, 0x00, 0x00, 
  0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x03, 0x80, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 
  0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x01, 0x86, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xcc, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
// 'Frame2', 80x32px
const unsigned char Frame2 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x70, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x03, 0x80, 0x00, 0x00, 0x00, 0x00, 
  0x03, 0x80, 0x00, 0x00, 0x0f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x01, 0xc0, 0x00, 0x00, 0x08, 0x18, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x18, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 
  0x00, 0x00, 0x18, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0xdf, 0xf8, 0x00, 0x08, 0x66, 0x00, 0x00, 
  0x00, 0x00, 0x01, 0xb8, 0xff, 0xff, 0xfc, 0x02, 0x00, 0x00, 0x00, 0x00, 0x01, 0xb0, 0x01, 0xfe, 
  0x06, 0x02, 0x00, 0x00, 0x00, 0x00, 0x03, 0x60, 0x00, 0x32, 0x03, 0x86, 0x00, 0x00, 0x00, 0x03, 
  0xff, 0xc0, 0x00, 0x33, 0x80, 0xfc, 0x00, 0x00, 0x00, 0x07, 0x07, 0x80, 0x00, 0x10, 0xe0, 0x01, 
  0x00, 0x00, 0x00, 0x7f, 0xff, 0x00, 0x00, 0x10, 0x38, 0x03, 0x00, 0x00, 0x00, 0x7f, 0xfe, 0x00, 
  0x00, 0x18, 0x0f, 0x86, 0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00, 0x08, 0x00, 0xfc, 0x00, 0x00, 
  0x01, 0x80, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x0c, 
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x01, 0x8c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00
};
// 'Frame3', 80x32px
const unsigned char Frame3 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x01, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x0e, 0x00, 0x00, 0x07, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x80, 0x00, 0x0c, 0x30, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xc0, 0x00, 0x10, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 
  0xf0, 0x00, 0x10, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xbe, 0x00, 0x18, 0x66, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x07, 0x87, 0xc0, 0x08, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0xff, 
  0xfc, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x1f, 0x06, 0x02, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x0e, 0x00, 0x09, 0x03, 0x86, 0x00, 0x00, 0x00, 0x00, 0x3f, 0xfe, 0x00, 0x09, 0x00, 0xfc, 
  0x00, 0x00, 0x00, 0x00, 0x6f, 0xfc, 0x00, 0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x58, 0x00, 
  0x00, 0x10, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x10, 0x80, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0xc0, 0x00, 0x00, 0x30, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0x20, 
  0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x1c, 0x60, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x60, 0x07, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00
};
// 'Frame4', 80x32px
const unsigned char Frame4 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x80, 0x00, 0x07, 0xe0, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x0c, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x3f, 0x00, 0x18, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0xe0, 0x10, 0x66, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x0c, 0x3e, 0x10, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x07, 
  0xf0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0xd8, 0x02, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x0a, 0x03, 0xce, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x06, 0x83, 0x84, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7e, 0x0d, 0x80, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 
  0xcc, 0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xf8, 0x61, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x0f, 0x80, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x03, 0x82, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x0c, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x02, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x60, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78, 0x00, 0xe0, 0x00, 0x00, 0x00
};
// 'Frame5', 80x32px
const unsigned char Frame5 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0xf8, 
  0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0xff, 0x0c, 0x06, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 
  0x03, 0x83, 0xc8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x06, 0x00, 0xf8, 0x19, 0x00, 0x00, 
  0x00, 0x00, 0x01, 0xc0, 0x0c, 0x00, 0xd8, 0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x70, 0x18, 0x00, 
  0x88, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xf8, 0x00, 0x8e, 0x00, 0x80, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x1c, 0x01, 0x83, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x01, 0x00, 0xfc, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0xc7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x02, 0x5e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x56, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x01, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xe4, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x8c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x08, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x0c, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00
};
// 'Frame6', 80x32px
const unsigned char Frame6 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61, 0xc0, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x01, 0x00, 0x30, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x19, 0x81, 0xc0, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x1f, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 
  0x00, 0x7e, 0x06, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0xd2, 0x06, 0x20, 0x00, 0x00, 
  0x00, 0x10, 0x00, 0x08, 0x01, 0x92, 0x00, 0x60, 0x00, 0x00, 0x00, 0x10, 0x00, 0x08, 0x01, 0x33, 
  0x00, 0xc0, 0x00, 0x00, 0x00, 0x18, 0x00, 0x18, 0x03, 0x61, 0xe3, 0x00, 0x00, 0x00, 0x00, 0x0c, 
  0x00, 0x10, 0x03, 0xc0, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x1f, 0x86, 0xc0, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x03, 0x00, 0x7c, 0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xc1, 0xc7, 
  0xe5, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x1b, 0x99, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0xf3, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x37, 0xc6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0xdb, 0xec, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xb8, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 
  0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00
};
// 'Frame7', 80x32px
const unsigned char Frame7 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 
  0xe0, 0x7f, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x79, 0x80, 0x40, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x01, 0xc0, 0x1d, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x3f, 0x03, 
  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x67, 0x03, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x0c, 0x00, 0xc5, 0x80, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x01, 0x84, 0xc0, 0x60, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x18, 0x03, 0x04, 0x7f, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x02, 
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x30, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x70, 0x00, 0xf0, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x0f, 0xfc, 0x01, 0x8c, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x01, 0xf8, 0x3e, 0x00, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x37, 0x00, 0x98, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x03, 0x10, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x21, 0x86, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xc0, 
  0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x43, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x13, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x32, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
// 'Frame8', 80x32px
const unsigned char Frame8 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x08, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x06, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xf0, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xe3, 0x30, 0x03, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x1f, 0x04, 0x18, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x18, 0x1f, 
  0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xc0, 0x60, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x0f, 0x00, 0xc0, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0xe0, 0x00, 0x80, 0x00, 
  0x00, 0x00, 0x03, 0xff, 0xf8, 0x00, 0x38, 0x01, 0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0xf0, 0x00, 
  0x1e, 0x01, 0x00, 0x00, 0x00, 0x00, 0x78, 0x01, 0xf0, 0x00, 0x03, 0xc3, 0x00, 0x00, 0x00, 0x00, 
  0xc0, 0x03, 0x30, 0x00, 0x00, 0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x30, 0x00, 0x01, 0x02, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x20, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 
  0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0xe0, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x80, 0x20, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x0c, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00, 0x20, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 
  0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x3f, 0x80, 0x00, 0x00, 0x00, 0x00
};
// 'Frame9', 80x32px
const unsigned char Frame9 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0xf0, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x10, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x32, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x7f, 0xff, 0xf8, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0xe0, 0x03, 0x88, 
  0x02, 0x00, 0x00, 0x01, 0xff, 0x00, 0x3e, 0x00, 0x02, 0xc4, 0x02, 0x00, 0x00, 0x0f, 0x01, 0xf8, 
  0xe0, 0x00, 0x06, 0x62, 0x02, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xc0, 0x00, 0x04, 0x39, 0xfc, 0x00, 
  0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x0c, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 
  0x07, 0xc1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x78, 0x80, 0x00, 0x00, 0x00, 
  0x00, 0x6c, 0x00, 0x00, 0x00, 0x0f, 0x80, 0x00, 0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00, 0x01, 
  0x40, 0x00, 0x00, 0x00, 0x03, 0x98, 0x00, 0x00, 0x00, 0x01, 0x40, 0x00, 0x00, 0x00, 0x0e, 0x30, 
  0x00, 0x00, 0x00, 0x01, 0x60, 0x00, 0x00, 0x00, 0xf8, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x01, 0x81, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
// 'Frame10', 80x32px
const unsigned char Frame10 [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xe0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x00, 
  0x00, 0x0c, 0x18, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x00, 0x08, 0x04, 0x00, 0x00, 0x00, 
  0x1e, 0x00, 0x1f, 0xff, 0xf0, 0x10, 0x62, 0x00, 0x00, 0x00, 0x03, 0xc1, 0xf8, 0x00, 0x1f, 0xf8, 
  0x62, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x01, 0xd8, 0x02, 0x00, 0x00, 0x00, 0x00, 0x78, 
  0x00, 0x00, 0x00, 0xc4, 0x02, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x63, 0x84, 0x00, 
  0x00, 0x00, 0x01, 0xc0, 0x00, 0x00, 0x00, 0x70, 0xf8, 0x00, 0x00, 0x03, 0xff, 0x80, 0x00, 0x00, 
  0x00, 0x0c, 0xc0, 0x00, 0x00, 0x1c, 0x06, 0x00, 0x00, 0x00, 0x00, 0x06, 0x3e, 0x00, 0x00, 0x30, 
  0x1c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x01, 0x80, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x00, 0x01, 
  0x80, 0x40, 0x00, 0x03, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x40, 0x00, 0x0e, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x00, 
  0x03, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// 'Marios_ideas', 128x32px
const unsigned char Logo [] PROGMEM = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x03, 0xce, 0x00, 0x00, 0x1e, 0x00, 0x04, 0x00, 0x1c, 0x7f, 0x8f, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x07, 0xff, 0x00, 0x00, 0x0c, 0x00, 0x0e, 0x00, 0x1c, 0x7f, 0x8f, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x0f, 0xff, 0x80, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x1c, 0x7f, 0x8f, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x0e, 0x73, 0x87, 0x81, 0xdc, 0x1e, 0x08, 0x0c, 0x1c, 0x7c, 0x8f, 0x0f, 0xc3, 0xfc, 0xff, 
  0x00, 0x0e, 0x73, 0x8f, 0xc3, 0xdc, 0x7f, 0x00, 0x1c, 0x1c, 0x78, 0x8e, 0x07, 0x81, 0xf8, 0xff, 
  0x00, 0x0e, 0x73, 0x9f, 0xe3, 0x9c, 0x7f, 0x00, 0x3c, 0x1c, 0x70, 0x8c, 0x03, 0x00, 0xf0, 0xff, 
  0x00, 0x0e, 0x73, 0x9c, 0x63, 0x1c, 0xe3, 0x80, 0x3c, 0x1c, 0x71, 0x8c, 0x63, 0x1c, 0xf0, 0xff, 
  0x00, 0x0e, 0x73, 0x98, 0x63, 0x1c, 0xe3, 0x80, 0x38, 0x1c, 0x73, 0x8c, 0xc3, 0x3c, 0xf3, 0xff, 
  0x00, 0x0e, 0x73, 0x9c, 0x63, 0x1c, 0xe3, 0x80, 0x38, 0x1c, 0x71, 0x8c, 0x7f, 0x1c, 0xf3, 0xff, 
  0x00, 0x0e, 0x73, 0x9f, 0x63, 0x1c, 0x7f, 0x00, 0xf8, 0x1c, 0x70, 0x1c, 0x3f, 0x0c, 0xc3, 0xff, 
  0x00, 0x0e, 0x73, 0x8f, 0x63, 0x1c, 0x7f, 0x00, 0xf0, 0x1c, 0x78, 0x1e, 0x1f, 0x8c, 0xc7, 0xff, 
  0x00, 0x0e, 0x73, 0x87, 0x63, 0x1c, 0x1c, 0x00, 0xe0, 0x1c, 0x7c, 0x7f, 0x0f, 0xcc, 0xcf, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

// 'Marios_ideas', 128x32px
const unsigned char Logo1 [] PROGMEM = {
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xfc, 0x31, 0xff, 0xff, 0xe1, 0xff, 0xfb, 0xff, 0xe3, 0x80, 0x70, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xf8, 0x00, 0xff, 0xff, 0xf3, 0xff, 0xf1, 0xff, 0xe3, 0x80, 0x70, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xf0, 0x00, 0x7f, 0xff, 0xff, 0xff, 0xe3, 0xff, 0xe3, 0x80, 0x70, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xf1, 0x8c, 0x78, 0x7e, 0x23, 0xe1, 0xf7, 0xf3, 0xe3, 0x83, 0x70, 0xf0, 0x3c, 0x03, 0x00, 
  0xff, 0xf1, 0x8c, 0x70, 0x3c, 0x23, 0x80, 0xff, 0xe3, 0xe3, 0x87, 0x71, 0xf8, 0x7e, 0x07, 0x00, 
  0xff, 0xf1, 0x8c, 0x60, 0x1c, 0x63, 0x80, 0xff, 0xc3, 0xe3, 0x8f, 0x73, 0xfc, 0xff, 0x0f, 0x00, 
  0xff, 0xf1, 0x8c, 0x63, 0x9c, 0xe3, 0x1c, 0x7f, 0xc3, 0xe3, 0x8e, 0x73, 0x9c, 0xe3, 0x0f, 0x00, 
  0xff, 0xf1, 0x8c, 0x67, 0x9c, 0xe3, 0x1c, 0x7f, 0xc7, 0xe3, 0x8c, 0x73, 0x3c, 0xc3, 0x0c, 0x00, 
  0xff, 0xf1, 0x8c, 0x63, 0x9c, 0xe3, 0x1c, 0x7f, 0xc7, 0xe3, 0x8e, 0x73, 0x80, 0xe3, 0x0c, 0x00, 
  0xff, 0xf1, 0x8c, 0x60, 0x9c, 0xe3, 0x80, 0xff, 0x07, 0xe3, 0x8f, 0xe3, 0xc0, 0xf3, 0x3c, 0x00, 
  0xff, 0xf1, 0x8c, 0x70, 0x9c, 0xe3, 0x80, 0xff, 0x0f, 0xe3, 0x87, 0xe1, 0xe0, 0x73, 0x38, 0x00, 
  0xff, 0xf1, 0x8c, 0x78, 0x9c, 0xe3, 0xe3, 0xff, 0x1f, 0xe3, 0x83, 0x80, 0xf0, 0x33, 0x30, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};



void setup() {

  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }

  // Show initial display buffer contents on the screen --
  // the library initializes this with an Adafruit splash screen.
  display.display();
  delay(2000); // Pause for 2 seconds

  // Displaying My Channel Logo
  // Displaying 4x Positive and Negative of the Channel Logo in sequence 
  
  for (int i=1 ; i<5; i++){
    display.clearDisplay();
    display.drawBitmap(0,0,Logo, 128, 32, 1);
    display.display();
    delay(500);
    display.clearDisplay();
    display.drawBitmap(0,0,Logo1, 128, 32, 1);
    display.display();
    delay(500);
  } 
}

void loop() {
  
  // Diplay Animation
  
  // Frame1
  display.clearDisplay();
  display.drawBitmap(30,0,Frame1, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame2
  display.clearDisplay();
  display.drawBitmap(30,0,Frame2, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame3
  display.clearDisplay();
  display.drawBitmap(30,0,Frame3, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame4
  display.clearDisplay();
  display.drawBitmap(30,0,Frame4, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame5
  display.clearDisplay();
  display.drawBitmap(30,0,Frame5, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame6
  display.clearDisplay();
  display.drawBitmap(30,0,Frame6, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame7
  display.clearDisplay();
  display.drawBitmap(30,0,Frame7, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame8
  display.clearDisplay();
  display.drawBitmap(30,0,Frame8, 80, 32, 1);
  display.display();
  delay(frame_delay);
  
  // Frame9
  display.clearDisplay();
  display.drawBitmap(30,0,Frame9, 80, 32, 1);
  display.display();
  delay(frame_delay); 
  
  // Frame10
  display.clearDisplay();
  display.drawBitmap(30,0,Frame10, 80, 32, 1);
  display.display();
  delay(frame_delay); 

  if (frame_delay>100) frame_delay=frame_delay-50;
  
}