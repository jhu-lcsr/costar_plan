

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


## Dynamics

There is a set of default "dynamics" provided both for simulated and physical robots, but these will not be able to handle all arbitrary robots and grippers. Consider making a `SimulatedDynamics` and `SubsccriberDynamics` of your own.

Note that the `SimulatedDynamics` is the set of dynamics that the planner uses to advance the world state; this is not a set of dynanics used for Gazebo or something!
