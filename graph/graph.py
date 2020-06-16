#GRAPH PART
import serial
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import  animation
import pandas as pd


data = serial.Serial('COM1',9600)

leng = 200
fig = plt.figure(figsize=(12, 9))

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

graphX, = ax1.plot([], [], 'b', label = 'X')
graphY, = ax2.plot([], [], 'r', label = 'Y')
graphZ, = ax3.plot([], [], 'g', label = 'Z')
axes = [ax1, ax2, ax3]

for ax in axes:
    ax.set_xlim(0, leng-1)
    ax.set_ylim(-2, 2)
    ax.set_ylabel('Acceleration [G]')
    ax.legend(loc='upper right')
    ax.grid(True)

ax1.set_title('Real-time sensor data')
ax3.set_xlabel('Data points')

t = list(range(0, leng))
accX = []
accY = []
accZ = []
gyroX = []
gyroY = []
gyroZ = []

for i in range(0, leng):
    accX.append(0)
    accY.append(0)
    accZ.append(0)

def init():
    graphX.set_data([], [])
    graphY.set_data([], [])
    graphZ.set_data([], [])
    return graphX, graphY, graphZ



def convert_float(value):
    try:
        return float(value)
    except ValueError:
        #print ('Data {} is not in float and hence marked to zero'.format(value))
        return 0.0

def animate(i):
    global t, accX, accY, accZ

    while (data.inWaiting() == 0):
        pass

    arduinoString = data.readline().decode("utf-8")
    dataArray = arduinoString.split(',')
    print(dataArray)
    accX.append(float(dataArray[0])/(32767/2))    
    accY.append(float(dataArray[1])/(32767/2))    
    accZ.append(float(dataArray[2])/(32767/2))
    accX.pop(0)
    accY.pop(0)
    accZ.pop(0)

    graphX.set_data(t, accX)
    graphY.set_data(t, accY)
    graphZ.set_data(t, accZ)

    return graphX, graphY, graphZ

delay = 20
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               interval=delay, blit=True)

plt.show()
#ser.close()
    
