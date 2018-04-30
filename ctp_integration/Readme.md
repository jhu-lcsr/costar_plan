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
