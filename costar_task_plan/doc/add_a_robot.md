

# How to add a robot

This is a guide on how the interface with ROS actually works.

## World Updates

### Topic Listeners

In general, the ROS interface is provided via the `CostarWorld` base class.

#### Gripper Listener

Anything that takes gripper input needs to implement the `AbstractGripperListener` base class. This class consumes a message via the `callback` function, and returns a status code via the `getStatus` function.

To summarize:
  - `gripper_listener.callback(msg)` -- ROS subscriber callback
  - `gripper_listener.getStatus(msg)` -- called by `CostarWorld` to update the actor.

#### Joint Listener

This is a pretty simple interface that captures the robot's last observed joint position.

### Object Poses

These read in through TF.

