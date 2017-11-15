
# Data Collection with CoSTAR

This data collection pipeline is designed to create datasets for training with the CTP library and its tools for object manipulation tasks. 

## Prerequisites

You will need access to the (costar_stack)[http://github.com/cpaxton/costar_stack] library in order to run any of these examples. For more instructions, check out the CoSTAR installation guide.

For installation issues or questions, email me (cpaxton at jhu.edu).

## ROS Topics

To verify your data set:

```
rosbag record /joint_states \
  /camera/depth_registered/points 
  /camera/depth/camera_info \
  /camera/rgb/camera_info 
  /costar/messages/info
```
