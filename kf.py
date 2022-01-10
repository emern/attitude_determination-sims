"""
Extended Kalman filter simulation in Python using default BMX160 Arduino driver

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
from numpy.linalg import inv
import time

sample_rate = 0.005 #less than Nyquist cutoff (source at 50ms sample time)
rc_cutoff_m = 300 #300 rad/s = ~48hz
rc_cutoff_g = 300 #300 rad/s = ~48hz
rc_cutoff_a = 300 #300 rad/s = ~48hz

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
        self.state = np.array([[0],
                                [0],
                                [0]])
        self.state_est = np.array([[0],
                                [0],
                                [0]])
        self.P = np.identity(3)
        self.P_est = np.identity(3)

# Device state vector instance
global_state = state()

#sensor data objects for mag, gyro and accel
mag = sensor_data()
gyro = sensor_data()
accel = sensor_data()

# List objects for plotting
measured_accel_mag_roll = list()
measured_accel_mag_pitch = list()
measured_accel_mag_yaw = list()

state_estimate_roll = list()
state_estimate_pitch = list()
state_estimate_yaw = list()

state_found_roll = list()
state_found_pitch = list()
state_found_yaw = list()

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
        data.current[i] = float(data.raw[i])
        data.last[i] = data.current[i]

# Body rates to euler angle rates transformation matrix:
# http://www.stengel.mycpanel.princeton.edu/Quaternions.pdf
def R_trans_mat(phi, theta):
    return np.array([[1, m.sin(phi) * m.tan(theta), m.cos(phi) * m.tan(theta)],
                    [0, m.cos(phi) , -m.sin(phi)],
                    [0, m.sin(phi) * (1/m.cos(theta)), m.cos(phi) * (1/m.cos(theta))]])

# Jacobian matrix (A)
# This has been precalculated and could probably be wrong
def jac_a(q, r, phi, theta):

    return np.array([[-m.tan(theta) * ((r * m.sin(phi)) - (q * m.cos(phi))), ((1/m.cos(theta))**2) * (q * m.sin(phi) + r * m.cos(phi)), 0],
                    [-q * m.sin(phi) - r * m.cos(phi), 0, 0],
                    [(q * m.cos(phi) - r * m.sin(phi)) / (m.cos(theta)), ((q * m.sin(phi) + r * m.cos(phi)) * m.sin(theta)) / ((m.cos(theta) ** 2)), 0]])

# Jacobian matrix (C) (normalized)
# This has been precalculated and could probably be wrong
def jac_c(phi, theta):
    return np.array([[0, 0, 1], # Magnetometer yaw
                    [0, -m.cos(theta), 0], # Ax
                    [m.cos(theta) * m.cos(phi), -m.sin(theta) * m.sin(phi), 0], #Ay
                    [-m.cos(theta) * m.sin(phi), -m.cos(phi) * m.sin(theta), 0]]) #Az



# Gets IMU transformation matrix (or DCM) which can be used to find euler angles
# For this filter, the direction "north" is represented as x, "east" as y and "down" as z
# Based of methodology shown here: https://www.youtube.com/watch?v=0rlvvYgmTvI
def accel_mag(mag, accel):

    # Find interial frame with accel and mag

    # Find accel and mag unit vectors
    mag_v = np.array([mag.current[0], mag.current[1], mag.current[2]])
    acc_v = np.array([accel.current[0], accel.current[1], accel.current[2]])

    mag_v_hat = mag_v / np.linalg.norm(mag_v)
    acc_v_hat = acc_v / np.linalg.norm(acc_v)

    # All vectors must be unit vectors

    return acc_v_hat, mag_v_hat


def sensor_model_output(acc_v_hat, mag_v_hat):
    phi = m.atan2(acc_v_hat[1], acc_v_hat[2])
    theta = m.atan2(-acc_v_hat[0], (m.sqrt((acc_v_hat[1] ** 2) + (acc_v_hat[2] ** 2))))
    m_corr_x = m.cos(theta) * mag_v_hat[0] + m.sin(phi) * m.sin(theta) * mag_v_hat[1] + m.cos(phi) * m.sin(theta) * mag_v_hat[2]
    m_corr_y = m.cos(phi) * mag_v_hat[1] - m.sin(phi) * mag_v_hat[2]

    return np.array([[m.atan2(-m_corr_y, m_corr_x)],
                    [acc_v_hat[0]],
                    [acc_v_hat[1]],
                    [acc_v_hat[2]]])

def sensor_model_output_only_angles(acc_v_hat, mag_v_hat):
    phi = m.atan2(acc_v_hat[1],  acc_v_hat[2])
    theta = m.atan2(-acc_v_hat[0],  (m.sqrt((acc_v_hat[1] ** 2) + (acc_v_hat[2] ** 2))))
    m_corr_x = m.cos(theta) * mag_v_hat[0] + m.sin(phi) * m.sin(theta) * mag_v_hat[1] + m.cos(phi) * m.sin(theta) * mag_v_hat[2]
    m_corr_y = m.cos(phi) * mag_v_hat[1] - m.sin(phi) * mag_v_hat[2]

    return np.array([[phi],
                    [theta],
                    [m.atan2(-m_corr_y, m_corr_x)]])

def sensor_model_prediction(phi, theta, epsilon):
    return np.array([[float(epsilon)], # Yaw_mag
                    [-m.sin(theta)], # Ax
                    [m.cos(theta) * m.sin(phi)], # Ay
                    [m.cos(theta) * m.cos(phi)]]) # Az


# Find gyro angles based on previous data and current rate
# Raw gyro data gives values in body frame rotational rates, need to transform into rotational euler rates and integrate
# This relies on current state estimate
# X[n] = X[n-1] + T * dX[n]
# Function will also return Jacobian matrix "A"
def find_gyro_angles(time_step, gyro, phi_est, theta_est, psi_est):

    # Filter data
    rc_filter(rc_cutoff_g, gyro)

    # Find body frame rates

    # p represents body frame rotation around x axis
    p_dot = gyro.current[0] * (np.pi / 180)

    # q represents body frame rotation around y axis
    q_dot = gyro.current[1] * (np.pi / 180)

    # r represents body frame rotation around z axis
    r_dot = gyro.current[2] * (np.pi / 180)

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

    # Generate A jacobian based on current rate and estimate
    jac_A_new = jac_a(q_dot, r_dot, phi_new, theta_new)

    out_vec = np.array([[float(phi_new)],
                        [float(theta_new)],
                        [float(psi_new)]])

    return jac_A_new, out_vec

# TRIAD output model
# Can be found in figure 3 of https://robomechjournal.springeropen.com/articles/10.1186/s40648-020-00185-y
def triad_output_model(phi, theta, epsilon):
    return np.array([[m.atan2(-m.sin(theta), m.cos(theta) * m.sin(phi))],
                    [m.atan2(m.sin(theta), m.sqrt(((m.cos(theta) * m.sin(phi)) ** 2) + ((m.cos(theta) * m.cos(phi)) ** 2)))],
                    [m.atan2(m.sin(epsilon) * m.cos(theta), m.cos(epsilon) * m.cos(theta))]])

# EKF predict stage
# Basic math derived from Phils Lab video series on sensor fusion:
# https://www.youtube.com/watch?v=hQUkiC5o0JI&t=613s
def ekf_predict(curr_state, gyro, time_step, err_covariance, Q):

    # Find gyro based prediction and Jacobian matrix
    jacobian, prediction = find_gyro_angles(time_step, gyro, curr_state[0], curr_state[1], curr_state[2])

    # Derive estimation for error covariance matrix (P)
    # P[n+1] = P[n] + T * (A*P[n] + P[n]*At + Q)
    P_1 = np.matmul(jacobian, err_covariance)
    P_2 = np.matmul(err_covariance, jacobian.transpose())
    P_3 = np.add(P_1, np.add(P_2, Q))
    P_4 = time_step * P_3

    err_covariance_predict = np.add(err_covariance, P_4)

    return prediction, err_covariance_predict



# EKF update stage
# Basic math derived from Phils Lab video series on sensor fusion:
# https://www.youtube.com/watch?v=hQUkiC5o0JI&t=613s
def ekf_update(mag, accel, prediction, err_covariance_predict, R):

    # Find angles for normalized acceleration
    ac_hat, mag_hat = accel_mag(mag, accel)
    state_observation_measured = sensor_model_output(ac_hat, mag_hat)
    return_angles = sensor_model_output_only_angles(ac_hat, mag_hat)
    model_prediction = sensor_model_prediction(prediction[0], prediction[1], prediction[2])

    # Calculate Jacobian "C"
    C = jac_c(prediction[0], prediction[1])

    # Find Kalman gain (K)
    # K = P * C(transpose) * [C * P * C(transpose) + R]^(-1)
    k_1 = np.matmul(err_covariance_predict, C.transpose())
    k_2 = np.matmul(C, err_covariance_predict)
    k_3 =  np.matmul(k_2, C.transpose())
    k_4 = np.add(k_3, R)
    k_4_inv = inv(k_4)

    K_tot = np.matmul(k_1, k_4_inv)

    # Calculate state estimation
    # Estimated state = Prediction + K * (acceleromter - accelerometer_model)
    corr_1 = np.subtract(state_observation_measured, model_prediction)
    corr_2 = np.matmul(K_tot, corr_1)
    corrected_output = np.add(prediction, corr_2)

    # Update P for next step
    # P = (I - K * C) * P
    P_1 = np.matmul(K_tot, C)
    P_2 = np.subtract(np.identity(3), P_1)
    P_final = np.matmul(P_2, err_covariance_predict)

    return corrected_output, P_final, return_angles


# Extended Kalman filter execution loop
i=0
for i in range(1000):

    # Increasing this value will increase the accel/mag noise smoothing
    R = 1500 * np.identity(4)
    # Increasing this value will decrease the gyro noise smoothing
    Q = np.identity(3) * 1500

    # Initialize at time 0
    if i == 0:
        #collect complete datapacket
        collect_packet()
        # run filtering algorithm
        rc_filter(rc_cutoff_m, mag)
        rc_filter(rc_cutoff_g, accel)
        ac_hat, mag_hat = accel_mag(mag, accel)
        global_state.state = sensor_model_output_only_angles(ac_hat, mag_hat)

    # Prediction step
    global_state.state_est, global_state.P_est = ekf_predict(global_state.state, gyro, sample_rate, global_state.P, Q)

    # Sleep for the set sample rate T
    time.sleep(sample_rate)
    # Collect new packet
    collect_packet()
    # run filtering algorithm
    rc_filter(rc_cutoff_m, mag)
    rc_filter(rc_cutoff_g, accel)


    # Update step

    global_state.state, global_state.P, measured = ekf_update(mag, accel, global_state.state_est, global_state.P_est, R)

    # Append data to plot
    measured_accel_mag_roll.append(measured[0])
    measured_accel_mag_pitch.append(measured[1])
    measured_accel_mag_yaw.append(measured[2])


    state_estimate_roll.append(global_state.state_est[0])
    state_estimate_pitch.append(global_state.state_est[1])
    state_estimate_yaw.append(global_state.state_est[2])

    state_found_roll.append(global_state.state[0])
    state_found_pitch.append(global_state.state[1])
    state_found_yaw.append(global_state.state[2])


    # Print for debugging
    print("")
    print("STATE")
    print(global_state.state)
    print("--ESTIMATE--")
    print(global_state.state_est)
    print("P")
    print(global_state.P)
    print("P_est")
    print(global_state.P_est)
    print("--")

print("DONE")

# Show plots at the end because matplotlib sucks and is slow :(

#set up and run animation
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)

ax1.plot(measured_accel_mag_roll, 'r', label='Accel/Mag Fusion Reading')
ax1.plot(state_estimate_roll, 'b', label='State Prediction')
ax1.plot(state_found_roll, 'g', label='Estimated Kalman State')
ax1.set_title("Roll axis")
#ax1.set_ylim([-2*np.pi, 2*np.pi])
ax1.legend()

ax2.plot(measured_accel_mag_pitch, 'r', label='Accel/Mag Fusion Reading')
ax2.plot(state_estimate_pitch, 'b', label='State Prediction')
ax2.plot(state_found_pitch, 'g', label='Estimated Kalman State')
ax2.set_title("Pitch axis")
#ax2.set_ylim([-2*np.pi, 2*np.pi])
ax2.legend()

ax3.plot(measured_accel_mag_yaw, 'r', label='Accel/Mag Fusion Reading')
ax3.plot(state_estimate_yaw, 'b', label='State Prediction')
ax3.plot(state_found_yaw, 'g', label='Estimated Kalman State')
ax3.set_title("Yaw axis")
#ax3.set_ylim([-2*np.pi, 2*np.pi])
ax3.legend()

plt.suptitle("1000 iterations of R = 1500, Q = 1500")
plt.show()

#close serial port
ser.close()