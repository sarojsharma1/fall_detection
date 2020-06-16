import serial
import numpy as np
import pandas as pd


data = serial.Serial('COM1',9600)


def convert_float(value):
    try:
        return float(value)
    except ValueError:
        #print ('Data {} is not in float and hence marked to zero'.format(value))
        return 0.0
    
df = pd.DataFrame()

while True:

    while(data.inWaiting() == 0):
        pass
    arduinoString = data.readline().decode("utf-8")    #string type
    dataArray = (arduinoString.split(','))             #dataArray list type/elements string type
    
    if len(dataArray)==6:
        dataArray = list(map(convert_float, dataArray)) #dataArray list type/elements float type
       
        #print(dataArray)

        ax1 = dataArray[0]
        ay1 = dataArray[1]
        az1 = dataArray[2]
        gx1 = dataArray[3]
        gy1 = dataArray[4]
        gz1 = dataArray[5]

        print(ax1,ay1,az1,gx1,gy1,gz1)

        #a = pd.DataFrame(dataArray).T
        #df = df.append(a)
        #print(df)     
           
               
    #else:
        #print ('There is some data format problem: {}'.format(dataArray))


