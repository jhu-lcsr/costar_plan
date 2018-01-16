# Husky Simulation

![Husky Navigation](camera_image.jpeg)

This example provides instructions on bringing up the husky robot simulation for data collection.

## Start Simulation

```
rosrun costar_simulation start --launch husky --experiment navigation
rosrun costar_simulation husky_test.py
```

## Old Simulation Version

To start the Husky demo the old way:

```
roslaunch husky_gazebo husky_playpen.launch
roslaunch husky_viz view_robot.launch
roslaunch husky_navigation exploration_demo.launch
rosrun costar_simulation husky_test.py
```
