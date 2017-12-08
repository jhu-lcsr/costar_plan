
# Generating a Task Model from Data

Sometimes we would like to create a task model that operates purely based on demonstration data, and can in fact connect many different things while expressing user preferences.

## Terminology

  - Action, option, activity: these are all different terms for the same word. In the remainder of this document we will use "action" to refer to a high-level, STRIPS-like action.
  - Control: reserved for low-level commands sent to PID controllers, not actual controls.

## Data Types

### Task Info

The Task Info message describes the transitions between high level actions, and is important when creating the graph of actions to execute.

Here is what the message looks like:
```
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
learning_planning_msgs/Transition[] transition
  string from
  string to
  int32 frequency
  learning_planning_msgs/TransitionParameters[] parameters
    std_msgs/Time start_time
      time data
    std_msgs/Time end_time
      time data
    string[] from_parameters
    string[] to_parameters
```

Each transition represents a transition in the CoSTAR "task" graph. The basis of this graph is the `Task` class, defined as:

```
from costar_task_plan.abstract import Task
task = Task() # Empty task
args = {} # for constructing option representation
task.add(to, from, args)
```

### Demonstration Info

Of the two different types of data, one contains all the information about a particular task. This is the Demonstration Info class.

The message definition is included below:
```
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
learning_planning_msgs/ObjectInfo[] object
  string name
  string object_class
  int64 id
  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z
    geometry_msgs/Quaternion orientation
      float64 x
      float64 y
      float64 z
      float64 w
learning_planning_msgs/HandInfo left
  string NO_OBJECT=none
  int8 GRIPPER_CLOSE=0
  int8 GRIPPER_OPEN=1
  int8 GRIPPER_FREE=2
  string activity
  string object_acted_on
  string object_in_hand
  int8 gripper_state
  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z
    geometry_msgs/Quaternion orientation
      float64 x
      float64 y
      float64 z
      float64 w
learning_planning_msgs/HandInfo right
  string NO_OBJECT=none
  int8 GRIPPER_CLOSE=0
  int8 GRIPPER_OPEN=1
  int8 GRIPPER_FREE=2
  string activity
  string object_acted_on
  string object_in_hand
  int8 gripper_state
  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z
    geometry_msgs/Quaternion orientation
      float64 x
      float64 y
      float64 z
      float64 w
```

There are three important categories:
  - the positions of all objects
  - the positions of the left and right hands
  - activity information
