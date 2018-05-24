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

# About collected datasets

Dataset files are saved to `~/.costar/data`. To view video from a dataset example use the following command:

```
python scripts/view_convert_dataset.py --preview True --path ~/.costar/data/2018-04-26-15-22-11_example000014.failure.h5f
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

The UR5 has quite a large Joint angle range, [-2pi, 2pi] on each joint,
but our URDF files which define the allowable range 
are configured to limit that to avoid unsafe or surprising motions.
However, if the robot gets rotated it can look like a reasonable position,
but to the software it looks dangerous and the robot will refuse to move.
We deliberately restrict some of the joints so the robot doesn't swing around dangerously.

When this happens you may encounter RRTConnect planning errors like the following: 
 
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

 
#### First thing to try

Look at the UR5 tablet, put the robot in teach mode by pushing the little black button on the tablet, 
and physically move the robot so all joints are centered (straight up).
#### Second thing to try

Try publishing a new arm joint goal to the ros driver directly which is at a supported location:

(note the commands below for this option may need some editing to work correctly, they are untested)

```
rostopic pub /joint_states sensor_msgs/JointState
header: 
  seq: 205208
  stamp: 
    secs: 1526066010
    nsecs: 184360042
  frame_id: ''
name: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
position: [-0.20010622784263976, -1.0068971277218561, -1.7816320343889247, -0.307711956239854, 1.4645159060792843, 1.6345561424432893]
velocity: [0.012783173237042155, -0.19907235682045749, 0.12785729871689563, -0.219670860426672, 0.032117725365165356, 0.15577096802105198]
effort: [0.34523884250831716, -0.7084121703417416, 3.185612955872199, 0.05185139998779967, 0.28670774110900993, 0.39498566461294454]

``` 

```
rostopic pub
/joint_traj_pt_cmd
header: 
  seq: 184206
  stamp: 
    secs: 1526065841
    nsecs: 616822641
  frame_id: ''
name: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
position: [-0.20059646892768868, -0.9814336154941703, -1.7995429410747648, -0.27933540890204817, 1.4603500816439703, 3.086226577293814]
velocity: [-1.2783173237042154e-05, 0.0, 0.0, 0.010066749741320484, 0.0, -0.34428284115316055]
effort: [-0.6052888797223742, 0.16365217859160489, 1.9571007111023433, 0.5673153175135729, -0.01220032940889404, 0.016775452937229304]

```

#### Final last resort workaround

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