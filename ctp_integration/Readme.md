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


## Running training on real data in an h5f
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
``