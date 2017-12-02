
# How to Collect Data for the Blocks Task

## Requirements

  - UR5 with robotiq 2-finger gripper running [CoSTAR](http://github.com/cpaxton/costar_stack)
  - Connected RGB-D camera supportd by CoSTAR (e.g. Primesense Carmine)

## Steps

### Script

This is the script that will actually start listening for new messages from ROS.

```
rosrun costar_task_plan collect_blocks_data.py
```

# Topics to record


    /camera/rgb/image_rect_color // rectified rgb color image
    /camera/rgb/camera_info // camera intrinsics and other matrices
    /camera/depth_registered/camera_info // depth registered camera intrinsics
    /camera/depth_registered/hw_registered/image_rect // rectified float depth image in meters
    /costar/messages/info // identifies different stages of task

camera_info topics will record:

 - matrices D,P,K,R
 - frame_id string
 - distortion_model string
 - int width
 - int height

## seriously consider recording

    /camera/depth_registered/points // rgbxyz point cloud, base frame is camera_rgb_optical_frame

# TF Transforms to record

All tf transforms should be recorded relative to base_link.

    camera_link
    camera_depth_frame
    camera_depth_optical_frame
    camera_rgb_frame
    camera_rgb_optical_frame // very important because this is the point cloud base frame
    table_frame
    ar_marker_0 (if visible)
    ar_marker_1 (if visible)
    ar_marker_2 (if visible)

## UR5 Transforms

    base_link // UR5 base, very important because everything can be done relative to this
    shoulder_link
    upper_arm_link
    wrist_1_link
    wrist_2_link
    wrist_3_link
    ee_link // very important because this is the wrist pose
    gripper_link
    gripper_center // very important because this is the point between the end of the gripper tips


# Notes

Reference code used to determine topics to use https://github.com/ros-drivers/rgbd_launch/blob/indigo-devel/launch/includes/depth_registered.launch.xml#L73

Consider encoding depth images with the google brain encoding, see depth_image_encoding.py function FloatArrayToRgbImage().
For decoding see GraspDataset._image_decode() for high performance decoding with tensors and ImageToFloatArray() for the slower numpy only version.

## Other topics to double check:

    /camera/depth/image_rect
    /camera/depth/camera_info
    /costar/DriverStatus
    /costar/detected_object_list
    /costar/display_trajectory
    /costar_perception/detected_object_list
    /costar_perception/visualized_cloud_topic
