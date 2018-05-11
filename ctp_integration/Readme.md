# CTP Integration

Code for testing Costar Task Planner on the real UR5 robot in our lab. This integrates with the [CoSTAR Stack](https://github.com/cpaxton/costar_stack).

## Quick Start

```
roslaunch ctp_integration bringup.launch
rosrun ctp_integration run.py --iter 1000
```

Quick start with logging and restarting upon crashes:
```
while true; do ./scripts/run.py --execute 1000 2>&1 | tee -a ctp_integration_run_log.txt; done
```

To view the costar_stack user interface:

```
roslaunch instructor_core instructor.launch
```

## Guidelines

  - python: contain specific source code
  - nodes: ROS nodes
  - launch: contains launch files


### Coding Guidelines

Not sure what ROS messages are or what fields are being set?

Try these command line commands:

```
rosmsg show 
rostpoic info
rosservice info
```

There is always a mapping from the ros messages to the code needed to fill out the messages.


## HowTo

### Run RVIZ visualization separately

To launch rviz separately, which helps with performance problems:

Open `costar_plan/ctp_integration/launch/bringup.launch` and set `rviz` to `false`.

To launch rviz manually run `roslaunch costar_bringup rviz.launch`.

### Running training on real data in an h5f
```
ahundt [5:27 PM]
@cpaxton do you have the command/config you used to train that first example of future prediction on real data?
cpaxton [5:34 PM]
is it not in the example?
ctp_model_tool --model conditional_image --features costar
probably a lot more flags but i dont remember them
that should be enough to get you started
oh
--data_file ./robot.h5f (edited)
```

## Troubleshooting

### The robot is running really slowly and reporting mysterious service and command errors

- If CPU usage is >77% ROS may not be able to keep up!
    - try killing rviz to if cpu usage is too high
- check if any of the most recently added code and configuration is adding delays

### The robot has stopped moving, help!

You may encounter RRTConnect planning errors like the following:

```
[ WARN] [1526008157.165810118]: Joint 'wrist_3_joint' from the starting state is outside bounds by a significant margin: [ -4.74226 ] should be in the range [ -3.14159 ], [ 3.14159 ] but the error above the ~start_state_max_bounds_error parameter (currently set to 0.1)
[ WARN] [1526008157.167005426]: Joint wrist_3_joint is constrained to be above the maximum bounds. Assuming maximum bounds instead.
[ WARN] [1526008157.167188742]: Joint wrist_3_joint is constrained to be above the maximum bounds. Assuming maximum bounds instead.

```

and the following:

```
[ WARN] [1526008157.169653829]: manipulator[RRTConnectkConfigDefault]: Skipping invalid start state (invalid bounds)
[ERROR] [1526008157.169698337]: manipulator[RRTConnectkConfigDefault]: Motion planning start tree could not be initialized!
[ WARN] [1526008157.169742859]: manipulator[RRTConnectkConfigDefault]: Skipping invalid start state (invalid bounds)
[ERROR] [1526008157.169777125]: manipulator[RRTConnectkConfigDefault]: Motion planning start tree could not be initialized!
[ WARN] [1526008157.169819083]: ParallelPlan::solve(): Unable to find solution by any of the threads in 0.000335 seconds
[ WARN] [1526008157.177523927]: Goal sampling thread never did any work.
[ INFO] [1526008157.177754869]: Unable to solve the planning problem
[WARN] [WallTime: 1526008157.180949] Done: -1
[INFO] [WallTime: 1526008157.181397] Planning returned code:-1
[ERROR] [WallTime: 1526008157.181647] DRIVER -- PLANNING failed

```

As you can see, somehow the robot thinks the `wrist_3_joint` is somewhere impossible!

Here is a super hacky workaround that will get around the joint angle planning error to get the robot moving again temporarily. 
Be sure to set it back ASAP and **do not commit this change** in the config files loaded for the robot
because it is probably not safe!

Open `ompl_planning_pipeline.launch.xml`, or its equivalent on your machine on the costar robot the specific file is here:

```
/home/cpaxton/catkin_ws/src/universal_robot/ur5_moveit_config/launch/ompl_planning_pipeline.launch.xml
```
Well I found a super risky looking setting that made the robot move again the setting below was previously 0.1 and I set it to 10, 

Find the following value:
```
  <arg name="start_state_max_bounds_error" value="10.0" />
```

And set it to:

```
  <arg name="start_state_max_bounds_error" value="10.0" />
```

Kill all the robot's rosnodes and fire it up again as stated in the Quick Start above.

### Something else is wrong and I'm not sure what to do!

Try the following ros command line commands:

 - `roswtf` - to see if there are any tf2 transform errors or node grap, and to  
 - `rosparam list` - to see what variables there are