
# ROS, Planning, and TOM

The TOM examples require the [TOM Robot package]() to execute.

## Downloading Dataset

The first dataset we have consists of a set of demonstrations of TOM picking and moving an "orange" from one place to another. These files are all available on Dropbox:
```
https://www.dropbox.com/sh/jucd681430959t2/AACGdPQp3z24VineOrYJSK4na?dl=0
```

Just download them and unpack into whatever location makes sense for you. You'll be running the CTP tool from the directory root after unpacking these data files.

## Getting started

Run the TOM simulator and server. This takes in joint positions and will move the arm to those positions:

```
roslaunch costar_task_plan tom.launch
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

## Execution

You can then execute with the command:

```
rosrun costar_task_plan tom_test.py --execute
```

This will publish trajectory goals to `joint_state_cmd`.

We expect objects will appear as TF frames.
