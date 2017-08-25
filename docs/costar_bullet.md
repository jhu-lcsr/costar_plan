# Getting Started with costar_bullet

## Basics

The easiest way to get started with costar is to type
``` rosrun costar_bullet start --task blocks --robot ur5 --agent task  --gui --show_images```
You will see the pybullet ExampleBrowser appear on your screen with a robotic arm that starts picking up objects.

**rosrun costar_bullet start** simply runs the **start** node from within the costar_bullet package. If you wish to
modify the setup of costar_plan then you can take a look
[here](https://github.com/cpaxton/costar_plan/tree/master/costar_bullet/scripts),
in particular at the **gui_start** program.

The ```--task``` tag refers to the task the arm is trying to complete. The task
in this case is called **blocks** and can be found [here](https://github.com/cpaxton/costar_plan/blob/c375e723bcdf65634253b6954076d0a41070ba71/costar_task_plan/python/costar_task_plan/simulation/tasks/blocks.py). While the
blocks task is the one that is at the furthest stage of development, other tasks like **obstructions** and **obstacles** are worth
taking a look at as well.

The **ur5** in the ```--robot ``` tag dictates the type of arm that will be used in the simulation. Several arms are currently available for development, however the ur5 arm is the only one that is officially supported at this time. Adding the necessary components and linking can be a bit complicated for integrating costar_plan with a different robotic arm and thus will be added to a separate guide shortly.

Finally the ```--gui``` tag just specifies that the gui window should appear on your screen. The gui window is most userful for debugging since with more complex scenarios it can be rather computationally intensive. Likewise the ``` show_images ``` tag - which displays the RGB, depth, and mask images for the arm - should also be used for debugging.

### Common errors

Remember to ``` source /opt/ros/indigo/setup.bash ``` (replace indigo with kinetic if you are using
Ubuntu 16.04) and to ``` source devel/setup.bash ``` inside your costar_ws directory. If you are experiencing errors with either
glewXInit or GL rendering, try updating your drivers if you are using an NVIDIA graphics card or using **gui_start** instead of **start**.


## Extending `costar_bullet` with new components

[Task Learning with CTP](docs/task_learning.md) shows how to use predictive models and other tools, this section is about how to integrate your own.

TODO(cpaxton) fill in the Extending costar bullet section

# Overview

## Key source files

`costar_plan/costar_task_plan/python/costar_task_plan/`:
#### abstract

#### agent
 - agent/ff.py
    - Main training loop

####  backend

#### datasets

#### grid_world

#### gym

#### mcts

#### models
this is where you create new learning models and algorithms that complete the specified tasks

#### needle_master

#### robotics

#### simulation
- /simulation/parse.py
    - Add new command line arguments to `costar_bullet`
- /simulation/option.py
- /simulation/world.py
    - Create world and access simulation world state
- simulation/features.py
    - Feature data like depth image, color image, arm joint angles, etc
    - Extracts feature data from the world for use in algorithms

#### tools

#### trainers


## Feature Creation

### Adding a feature to existing models

## Model Creation


# Task Creation

The goal of our hierarchical task learning is to be able to use symbolic high-level planning grounded by learned neural net models of the world.

Our training data is sequences of task execution created with the `costar_bullet` simulation tool, as per the figure below.

![Training Data](../photos/blocks_data.png)

The tool lets us generate large amounts of task performances, with randomized high level (and eventually low level) actions.

## Task Definition

The task is defined as a set of high- and low-level actions at various levels, given by a task plan such as that shown below. For now, we will consider the simple version of the "blocks" task.

Here, the robot can grab one of several blocks. Grabbing a block is divided between aligning, approaching, and closing the gripper. We wish to define a task architecture like this one:

![First stage of blocks task](../photos/blocks_task_1.png)

The code below defines this simplest version of the task, creating both the action generators themselves (called "options"), and associated conditions. There are three differnt things we need to create:
  - options: these are the high-level actions; each one is a "generator" from which we can sample different policies and termination conditions.
  - arguments: these contain an option constructor, arguments, and an optional field to remap from these arguments to a part of the option constructor.
  - task: the task model itself.

The key parts of the Option are the constructor and the `samplePolicy(world)` function, which takes a current world and returns a policy functor and a condition.

The block of code below, taken from [CTP's Blocks task](../costar_task_plan/python/costar_task_plan/simulation/tasks/blocks.py), defines the choice between different objects. Take a look at the [basic simulation options](../costar_task_plan/python/costar_task_plan/simulation/option.py) for examples of options and the `samplePolicy()` function.

``` python
AlignOption = lambda goal: GoalDirectedMotionOption(
    self.world,
    goal,
    pose=((0.05, 0, 0.05), self.grasp_q),
    pose_tolerance=(0.03, 0.025),
    joint_velocity_tolerance=0.05,)
align_args = {
    "constructor": AlignOption,
    "args": ["block"],
    "remap": {"block": "goal"},
}
GraspOption = lambda goal: GoalDirectedMotionOption(
    self.world,
    goal,
    pose=((0.0, 0, 0.0), self.grasp_q),
    pose_tolerance=(0.03, 0.025),
    joint_velocity_tolerance=0.05,)
grasp_args = {
    "constructor": GraspOption,
    "args": ["block"],
    "remap": {"block": "goal"},
}
LiftOption = lambda: GeneralMotionOption(
    pose=(self.over_final_stack_pos, self.grasp_q),
    pose_tolerance=(0.05, 0.025),
    joint_velocity_tolerance=0.05,)
lift_args = {
    "constructor": LiftOption,
    "args": []
}
PlaceOption = lambda: GeneralMotionOption(
    pose=(self.final_stack_pos, self.grasp_q),
    pose_tolerance=(0.05, 0.025),
    joint_velocity_tolerance=0.05,)
place_args = {
    "constructor": PlaceOption,
    "args": []
}
close_gripper_args = {
    "constructor": CloseGripperOption,
    "args": []
}
open_gripper_args = {
    "constructor": OpenGripperOption,
    "args": []
}
```

When we create a task model, we use the `add(name, parent, arg_dict)` function. Its three parameters are:
  - `name`: name of the option
  - `parent`: name of the predecessor, or `None` if predecessor is root
  - `arg_dict`: dictionary containing constructor function, list of argument names, and optionally a `remap` dictionary from argument names to option constructor parameters.

```
# Create a task model
task = Task()
task.add("align", None, align_args)
task.add("grasp", "align", grasp_args)
task.add("close_gripper", "grasp", close_gripper_args)
task.add("lift", "close_gripper", lift_args)
task.add("place", "lift", place_args)
task.add("open_gripper", "place", open_gripper_args)
task.add("done", "open_gripper", lift_args)
```

We automatically call the `task.compile()` function to create the task model.

# Creating a Data Set

You can create a small data set in the normal way:

```
rosrun costar_bullet start --robot ur5 --task blocks --agent task \
  --features multi --save -i 10 --data_file small.npz
```

It might be helpful to make a larger training data set than this one, which is possible by changing the number of iterations as seen below.

```
rosrun costar_bullet start --robot ur5 --task blocks --agent task \
  --features multi --save -i 100 --data_file bigger.npz
```


# Learning

We can now use the standard CoSTAR bullet tool to train a model:
```
rosrun costar_bullet start --robot ur5 --task blocks --agent null --features multi \
  -i 1000 --model hierarchical --data_file small.npz --load --si 5 --lr 0.001
```

Replace the options as necessary. Interesting things to change:
  - `--lr` sets the learning rate
  - `--optimizer` changes the optimizer (try `nadam`, `sgd`, etc.)
  - `--si` or `--show_images` changes how often we display results
  - `-i` is iterations for training the predictor...
  - `-e` is epochs for fitting other models (used with the Keras `model.fit()` function)

It takes several thousand iterations to get this right. An example at about 500 iterations:

![Example intermediate results](../photos/predictor_intermediate_results.png)

You should be able to see the model learning something useful fairly early on, at least with a small data set. It just won't be perfect.

# Testing

To run the basic model, no planning, use the same `ff` agent as you would for anything else:
```
rosrun costar_bullet start --robot ur5 --task blocks --agent ff --features multi \
  -i 1 --model hierarchical --load_model  --gui
```


### Integrating a Task, Model, and Features
