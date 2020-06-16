//https://howtomechatronics.com/tutorials/arduino/arduino-and-mpu6050-accelerometer-and-gyroscope-tutorial/
#include <Wire.h>
#include <math.h>
int16_t acc_x,acc_y,acc_z,temp,gyro_x,gyro_y,gyro_z;
float acc_vector,accx_error,accy_error,accz_error,gyrox_error,gyroy_error,gyroz_error;
const int MPU = 0x68;
int c = 0;

void setup() 
{
Serial.begin(9600);
Wire.setClock(400000);
Wire.begin();
Wire.beginTransmission(MPU);
Wire.write(0x6B);
Wire.write(0x00);
Wire.endTransmission(true);
//Calculate_IMU_error();
delay(5);
}


void loop() 
{
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU,14,true);
  acc_x = (Wire.read()<< 8 | Wire.read());
  acc_y = (Wire.read()<< 8 | Wire.read());
  acc_z = (Wire.read()<< 8 | Wire.read());
  temp = (Wire.read() << 8 | Wire.read());
  gyro_x = (Wire.read()<< 8 | Wire.read());
  gyro_y = (Wire.read()<< 8 | Wire.read());
  gyro_z = (Wire.read()<< 8 | Wire.read());
  Serial.print(acc_x);
  Serial.print(",");
  Serial.print(acc_y);
  Serial.print(",");
  Serial.print(acc_z);
  
  Serial.print(",");
  
  Serial.print(gyro_x);
  Serial.print(",");
  Serial.print(gyro_y);
  Serial.print(",");
  Serial.println(gyro_z);
 
  delay(0.05);                           /*200Hz*/
}


void Calculate_IMU_error()
{
  while(c < 200)
  {
    Wire.beginTransmission(MPU);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 14, true);
    acc_x = (Wire.read()<< 8 | Wire.read())/16384.0;
    acc_y = (Wire.read()<< 8 | Wire.read())/16384.0;
    acc_z = (Wire.read()<< 8 | Wire.read())/16384.0;
    
    accx_error = accx_error + ((atan((acc_y)/sqrt(pow((acc_x),2)+pow((acc_z),2)))*(180/PI)));
    accy_error = accy_error + ((atan(-1*(acc_x)/ sqrt(pow((acc_y),2)+pow((acc_z), 2))) * (180/PI)));
    
    temp = (Wire.read() << 8 | Wire.read());
    
    gyro_x = (Wire.read()<< 8 | Wire.read())/ 131.0;
    gyro_y = (Wire.read()<< 8 | Wire.read())/ 131.0;
    gyro_z = (Wire.read()<< 8 | Wire.read())/ 131.0;

    gyrox_error = gyrox_error + (gyro_x / 131.0);
    gyroy_error = gyroy_error + (gyro_y / 131.0);
    gyroz_error = gyroz_error + (gyro_z / 131.0);
    
    c++; 
  }
  
  accx_error = accx_error / 200;
  accy_error = accy_error / 200;

  gyrox_error = gyrox_error / 200;
  gyroy_error = gyroy_error / 200;
  gyroz_error = gyroz_error / 200;

  Serial.print(accx_error);  
}

  
