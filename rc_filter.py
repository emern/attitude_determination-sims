"""
Implements simple RC filtering in Python using default BMX160 Arduino driver

Required libs:
    pyserial
    numpy
    matplotlib
"""

import serial
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

sample_rate = 0.005 #less than Nyquist cutoff (source at 50ms sample time)
rc_cutoff = 50 #50 rad/s = ~8hz

# Plotting data
plt_step = 0
plt_list = list()
plt_mag_flt = list()
plt_accel_flt = list()
plt_gyro_flt = list()

plt_mag = list()
plt_accel = list()
plt_gyro = list()


#sensor data class
class sensor_data:
    def __init__(self):
        self.last = [0,0,0]
        self.current = [0,0,0]
        self.raw = list()
        self.found = False

#sensor data objects for mag, gyro and accel
mag = sensor_data()
gyro = sensor_data()
accel = sensor_data()

ser = serial.Serial()
ser.port="COM9" #enter your port here
ser.baudrate=115200

#open serial port
ser.open()

#Read a serial line from the arduino and return to host
def parse_line():
    resp_bytes = ser.readline()
    data = resp_bytes.decode("cp437")
    return data

#Subroutine to collect and process complete datapacket with built in error detection
def collect_packet():

    mag.found = False

    #For first iteration make sure that the first read line is the mag line
    while not mag.found:
        new_line = parse_line()
        if re.findall('M X: (-?\d+)  Y: (-?\d+)  Z: (-?\d+)', new_line):
            mag.raw = (list(re.findall('M X: (-?\d+)  Y: (-?\d+)  Z: (-?\d+)', new_line)[0]))
            mag.found = True

    #In the event that a line is skipped, recusively get next packet (Yes, this kind of breaks the filter, but this is a better option than crashing IMO)
    try:
        new_line = parse_line()
        gyro.raw = (list(re.findall('G X: (-?\d+)  Y: (-?\d+)  Z: (-?\d+)', new_line)[0]))
    except IndexError:
        del mag.raw[-1]
        collect_packet()

    #In the event that a line is skipped, recusively get next packet
    try:
        new_line = parse_line()
        accel.raw = (list(re.findall('A X: (-?\d+)  Y: (-?\d+)  Z: (-?\d+)', new_line)[0]))
    except IndexError:
        del gyro.raw[-1]
        del mag.raw[-1]
        collect_packet()

#based on TF of RC lowpass filter:
# (1-vo)/r = (vo/(1/cs))
# TF = 1/(1+r*c*s)
# Pole of TF at 1/(r*c)
# Must choose some constants R and C such that they give the desired lowpass
# We will discretize this using backwards euler method (dV/dt = (v[n] - v[n-1])/T)
# ODE for this equation becomes Vin = Vout + RCdVout/dt
def rc_filter(cutoff, data):
    c=0.1 #arbitrary
    r = 1/(cutoff * c) #calculate r for our filter

    #Vout = T/(T+RC)*Vin[n] + RC/(T+RC)Vin[n-1]
    for i in range(3):
        data.current[i] = float(data.raw[i]) * (sample_rate / (sample_rate + r*c)) + ((r*c/(sample_rate + r*c)) * data.last[i])
        data.last[i] = data.current[i]


#matplotlib animation frame. I hate this with every bit of my essence but matplotlib cannot handle being directly updated this quickly.
def animate(i):

    #import the number of plt steps
    global plt_step

    #collect complete datapacket
    collect_packet()

    #run filtering algorithm
    rc_filter(rc_cutoff, mag)
    rc_filter(rc_cutoff, gyro)
    rc_filter(rc_cutoff, accel)

    #append values to plot lists
    plt_mag.append(int(mag.raw[0]))
    plt_mag_flt.append(mag.current[0])

    plt_accel.append(int(accel.raw[0]))
    plt_accel_flt.append(accel.current[0])

    plt_gyro.append(int(gyro.raw[0]))
    plt_gyro_flt.append(gyro.current[0])
    plt_list.append(plt_step)

    #truncate plot lists if we have over 200 entries
    if len(plt_list) >= 200:
        del plt_list[0]
        del plt_mag[0]
        del plt_mag_flt[0]
        del plt_accel[0]
        del plt_accel_flt[0]
        del plt_gyro[0]
        del plt_gyro_flt[0]

    #plot figures
    ax1.clear()
    ax1.set_title("magnetometer x data")
    ax1.plot(plt_list, plt_mag)
    ax1.plot(plt_list, plt_mag_flt)

    ax2.clear()
    ax2.set_title("accelerometer x data")
    ax2.plot(plt_list, plt_accel)
    ax2.plot(plt_list, plt_accel_flt)

    ax3.clear()
    ax3.set_title("gyroscope x data")
    ax3.plot(plt_list, plt_gyro)
    ax3.plot(plt_list, plt_gyro_flt)

    #increment plot step
    plt_step +=1

# Configure animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

#set up and run animation
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ani = animation.FuncAnimation(fig, animate, interval=sample_rate, frames=500)
ani.save('animation_rcfilter.mp4',  writer = writer)

#close serial port
ser.close()