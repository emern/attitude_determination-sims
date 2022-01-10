"""
Complimentary filter simulation in Python using default BMX160 Arduino driver

Note: In order for this simulation to run, you must have an ffmpeg renderer installed on your system
    (https://www.wikihow.com/Install-FFmpeg-on-Windows)

Required libs:
    pyserial
    numpy
    matplotlib
"""

from numpy.core.numeric import base_repr
import serial
import re
import numpy as np
import math as m
from mpmath import sec
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

sample_rate = 0.005 #less than Nyquist cutoff (source at 50ms sample time)
rc_cutoff_m = 300 #300 rad/s = ~48hz
rc_cutoff_g = 300 #300 rad/s = ~48hz
rc_cutoff_a = 300 #300 rad/s = ~48hz

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

# Class for current state
class state:
    def __init__(self):

        # DCM array placeholder
        self.DCM = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        # Initialize current and last angles to 0
        self.euler = [0,0,0]
        self.euler_last = [0,0,0]

# Device state vector instance
global_state = state()

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

    # Vout = T/(T+RC)*Vin[n] + RC/(T+RC)Vin[n-1]
    for i in range(3):
        data.current[i] = float(data.raw[i]) * (sample_rate / (sample_rate + r*c)) + ((r*c/(sample_rate + r*c)) * data.last[i])
        data.last[i] = data.current[i]


# Euler rotation matrices
def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    return np.array([[ m.cos(theta), 0, m.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                    [ m.sin(theta), m.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

# Body rates to euler angle rates transformation matrix:
# http://www.stengel.mycpanel.princeton.edu/Quaternions.pdf
def R_trans_mat(phi, theta):
    return np.array([[1, m.sin(phi) * m.tan(theta), m.cos(phi) * m.tan(theta)],
                    [0, m.cos(phi) , -m.sin(phi)],
                    [0, m.sin(phi) * (1/m.cos(theta)), m.cos(phi) * (1/m.cos(theta))]])



# Gets IMU transformation matrix (or DCM) which can be used to find euler angles
# For this filter, the direction "north" is represented as x, "east" as y and "down" as z
# Based of methodology shown here: https://www.youtube.com/watch?v=0rlvvYgmTvI
def accel_mag(mag, accel):

    # run filtering algorithm, same cutoff for now
    rc_filter(rc_cutoff_m, mag)
    rc_filter(rc_cutoff_g, accel)

    # Find interial frame with accel and mag

    # Find accel and mag unit vectors
    mag_v = np.array([mag.current[0], mag.current[1], mag.current[2]])
    acc_v = np.array([accel.current[0], accel.current[1], accel.current[2]])

    mag_v_hat = mag_v / np.linalg.norm(mag_v)
    acc_v_hat = acc_v / np.linalg.norm(acc_v)

    # All vectors must be unit vectors

    # East (y) is cross product of downwards acceleration and magnetic field (mag points "north-down")
    y = np.cross(-acc_v_hat, mag_v_hat)
    y = y / np.linalg.norm(y)

    # North (x) is cross product of East (y) and down (z)
    x = np.cross(y, -acc_v_hat)
    x = x / np.linalg.norm(x)

    # Down (z) is negative accel z
    z = -acc_v_hat

    DCM = np.array([[x[0], y[0], z[0]],
                   [x[1], y[1], z[1]],
                   [x[2], y[2], z[2]]])
    return DCM


# Find euler angles from existing DCM
# Referanced from https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
# Also referanced from https://www.geometrictools.com/Documentation/EulerAngles.pdf
def find_angles_from_dcm(DCM):

    # phi = atan2(r21, r22)
    roll_phi = m.atan2(DCM[2, 1], DCM[2, 2])

    # theta  = atan2(-r20, sqrt(r21^2 + r22^2))
    pitch_theta = m.atan2(-DCM[2, 0], m.sqrt((DCM[2, 1] ** 2) + (DCM[2, 2] ** 2)))

    # Epsilon = atan2(r10, r00)
    yaw_psi = m.atan2(DCM[1, 0], DCM[0, 0])

    out_vec = np.array([[roll_phi],
                        [pitch_theta],
                        [yaw_psi]])

    return out_vec

# Find gyro angles based on previous data and current rate
# Raw gyro data gives values in body frame rotational rates, need to transform into rotational euler rates and integrate
# This relies on current state estimate
# X[n] = X[n-1] + T * dX[n]
def find_gyro_angles(time_step, gyro, phi_est, theta_est, psi_est):

    # Filter data
    rc_filter(rc_cutoff_g, gyro)

    # Find body frame rates

    # p represents body frame rotation around x axis
    p_dot = gyro.current[0]

    # q represents body frame rotation around y axis
    q_dot = gyro.current[1]

    # r represents body frame rotation around z axis
    r_dot = gyro.current[2]

    # Find the rate transformation DCM, http://www.stengel.mycpanel.princeton.edu/Quaternions.pdf
    DCM_rate = R_trans_mat(phi_est, theta_est)

    pqr = np.array([[p_dot],
                    [q_dot],
                    [r_dot]])

    # Find euler rates by multiplying
    euler_dot = np.matmul(DCM_rate, pqr)

    # euler_new = euler[n-1] + T * euler_dot (phils lab)
    phi_new = phi_est + euler_dot[0] * time_step

    theta_new = theta_est + euler_dot[1] * time_step

    psi_new = psi_est + euler_dot[2] * time_step

    # Convert to DCM for convenience
    DCM_new = np.matmul(np.matmul(Rz(psi_new), Ry(theta_new)), Rx(phi_new))

    # Euler angle output
    out_vec = np.array([[float(phi_new)],
                        [float(theta_new)],
                        [float(psi_new)]])

    return DCM_new, out_vec


#matplotlib animation frame. I hate this with every bit of my essence but matplotlib cannot handle being directly updated this quickly.
def animate(i):

    #collect complete datapacket
    print(i)
    collect_packet()

    global quiver_ext
    quiver_ext.remove()

    # "Instant" DCM, i.e rotation matrix generated with only accel and mag
    DCM_inst = accel_mag(mag, accel)

    # Extract instant values from DCM
    euler_inst = find_angles_from_dcm(DCM_inst)

    cols = ['r', 'g', 'b']

    if i == 0:
        # "Integrated values", i.e gyro integration over time to find angles
        DCM_int, euler_int = find_gyro_angles(sample_rate, gyro, euler_inst[0], euler_inst[1], euler_inst[2])
    else:
        DCM_int, euler_int = find_gyro_angles(sample_rate, gyro, global_state.euler_last[0],
                                                                global_state.euler_last[1], global_state.euler_last[2])

    # Sensor data weighting constant between 0 and 1. High alpha = high trust in accel/mag, low alpha = high trust in gyro
    # Conceptually, defines if we trust out integrated or instant value more.
    # Usually for drone/spacecraft situations we use very low alpha (just enough to remove some of the gyro noise/drift).
    # More info, Phils lab Ep.2 (https://www.youtube.com/watch?v=BUW2OdAtzBw)
    alpha = 0.10

    # Complimentary filter: Accel/Mag * alpha + (1-alpha) * gyro
    # Compute for both euler angles and DCM
    global_state.euler = np.add(euler_inst * alpha, (1-alpha) * euler_int)
    global_state.DCM = np.add(DCM_inst * alpha, (1-alpha) * DCM_int)

    # Plot DCM, removes issues of gimbal lock
    second_frame = np.array([[0,0,0, global_state.DCM[0, 0],
                                    global_state.DCM[0, 1],
                                    global_state.DCM[0, 2]],
                            [0,0,0,global_state.DCM[1, 0],
                                    global_state.DCM[1, 1],
                                    global_state.DCM[1, 2]],
                            [0,0,0,global_state.DCM[2, 0],
                                    global_state.DCM[2, 1],
                                    global_state.DCM[2, 2]]])

    X, Y, Z, U, V, W = zip(*second_frame)
    quiver_ext = ax.quiver(X, Y, Z, U, V, W, colors=cols, length=1, pivot='tail')

    # Euler angles used for axis labels
    ax.set_xlabel('x -> Roll: {}'.format(global_state.euler[0]))
    ax.set_ylabel('y -> Pitch: {}'.format(global_state.euler[1]))
    ax.set_zlabel('z -> Yaw: {}'.format(global_state.euler[2]))

    # Update system
    global_state.euler_last = global_state.euler



# Set up and run animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Needed for animation and 3d quiver plot
basis_vec = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
X, Y, Z, U, V, W = zip(*basis_vec)
quiver_base = ax.quiver(X, Y, Z, U, V, W, length=1)
quiver_ext = ax.quiver(X, Y, Z, U, V, W)

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Configure animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

# Run and save animation
ani = animation.FuncAnimation(fig, animate, interval=sample_rate, frames=300)
ani.save('animation.mp4',  writer = writer)

print("DONE")


#close serial port
ser.close()