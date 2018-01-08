# Guidelines

  - python: contain specific source code
  - nodes: ROS nodes
  - launch: contains launch files

# Start System

```
roslaunch ctp_tom planner.launch
```

# Collecting Segmentation Data

```
roslaunch ctp_tom collect_color_data.launch object:=orange
roslaunch ctp_tom collect_color_data.launch object:=blue_duplo
roslaunch ctp_tom collect_color_data.launch object:=yellow_duplo
roslaunch ctp_tom collect_color_data.launch object:=pink_duplo
roslaunch ctp_tom collect_color_data.launch object:=purple_duplo
```

