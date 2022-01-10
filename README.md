# attitude_determination-sims
Collection of simulations done as part of an investigation into satellite attitude determination algorithms for UBC Orbit.

## Requirements

* Python libs
 * Pyserial
 * Numpy
 * Matplotlib
* An ffmpeg renderer for saving simulations (https://www.wikihow.com/Install-FFmpeg-on-Windows)
* BMX160 IMU breakout connected to any sort of Arduino with DFRobot driver to collect raw data (https://wiki.dfrobot.com/BMX160_9_Axis_Sensor_Module_SKU_SEN0373)

## Simulations

There are 3 simulations included in this repository:

* rc_filter.py -> Smooth out raw sensor noise using digital analog of Resistor-Capacitor Low Pass Filter
* comp_filter.py -> Create an attitude estimation based on fused sensor data (using Complementary Filter)
* kf.py -> Create an attitude estimation based on fused sensor data (using Extended Kalman Filter)

