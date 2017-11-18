
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
