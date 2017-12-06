
# CTP on the Real TOM robot

## Starting

```
roslaunch ctp_tom planning.launch
```

## Components

### The Table Integrator

This script uses the [integrator tool](costar_task_plan/python/costar_task_plan/robotics/perception/transform_integrator.py) to publish a single frame for the table location in `/camera_rgb_optical_frame`.

The [table_integrator.py](ctp_tom/scripts/table_integrator.py) script itself just computes a few transforms based on observed data, and is not perfect.

You may wish to change:
  - the rotations and translations between markers. these are all transforms from the marker to `/ar_marker_0`. After this, an offset is applied to push everything into the right frame for the table.
  - the offset. This is what actually computes the location of the table.
  - history length. This is used to get "smooth" estimates of where the table is.

