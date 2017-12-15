
# ROS, Planning, and TOM

The TOM examples require the [TOM Robot package](https://github.com/cpaxton/tom_robot) to execute.

  - Installing Dependencies
  - Real Robot Setup
  - One-Arm Orange Tests

## Installing Dependencies

When using the real robot, we assume that you have access to a working version of the full [CoSTAR stack](https://github.com/cpaxton/costar_stack/). This contains perception code, data collection tools, et cetera, for running a robot and using our custom UI and other tools. You will not need all of this for TOM, so certain features can be freely disabled.

There is a prototype installation script, `install_tom.sh`, which should install parts of the CoSTAR stack and also CoSTAR plan.

### Disabled Packages

  - `costar_gazebo_plugins` does not support kinetic/Gazebo 7
  - `roboticsgroup_gazebo_plugins` does not support Gazebo 7
  - `robotiq_s_model_articulated_gazebo_plugins` does not support Gazebo 7
  - `sp_segmenter` requires opencv 2

## Real Robot Setup

There are a few different pieces that need to start up:
  - the planning context
  - the perception backend, including the Alvar AR tracker
  - the planner itself

You can start up the context and perception systems with the command:
```
roslaunch ctp_tom planning.launch
```

By default, the `real:=true` option is set. We also plan on supporting a fake version which creates a scene based on fixed TF poses. This does not simulate object interactions or anything fancy like that.

For more information, see the [real TOM guide](tom_real_robot.md).

## The Parser

The `ctp_tom` parser is a version of the task plan parsing tool that takes in messages and produces an executable task graph. You can feed the parser one or more rosbags:

```
# One bag only
rosrun ctp_tom parse.py --bagfile oranges_2017-12-13-19-01-15.bag
# Multiple bags
rosrun ctp_tom parse.py --bagfile oranges_2017-12-13-19-01-15.bag,blocks_2017-12-13-20-07-27.bag
```

Adding the `--fake` flag will add some fake objects and compile the model:

```
rosrun ctp_tom parse.py oranges_2017-12-13-19-01-15.bag,blocks_2017-12-13-20-07-27.bag --fake
```

## One-Arm Orange Tests (Old Version)

These are the "old" experiments, and may not work all that well any more.

### Downloading Dataset

The first dataset we have consists of a set of demonstrations of TOM picking and moving an "orange" from one place to another. These files are all available on Dropbox:
```
https://www.dropbox.com/sh/jucd681430959t2/AACGdPQp3z24VineOrYJSK4na?dl=0
```

Just download them and unpack into whatever location makes sense for you. You'll be running the CTP tool from the directory root after unpacking these data files.

### Getting started

Run the simple TOM simulator and server. This takes in joint positions and will move the arm to those positions:

```
roslaunch costar_task_plan simple_sim.launch
```

Then move into your data directory:
```
roscd costar_task_plan
cd data
rosrun costar_task_plan tom_test.py
```

We should see a trajectory appear, and output like:

```
0 / 11
1 / 11
2 / 11
3 / 11
4 / 11
5 / 11
6 / 11
7 / 11
8 / 11
9 / 11
10 / 11
11 / 11
=== close ===
1 / 11
1 / 11
2 / 11
3 / 11
4 / 11
5 / 11
6 / 11
7 / 11
8 / 11
9 / 11
10 / 11
11 / 11
=== open ===
```

### Execution

You can then execute with the command:

```
rosrun costar_task_plan tom_test.py --execute
```

This will publish trajectory goals to `joint_state_cmd`.

We expect objects will appear as TF frames.
