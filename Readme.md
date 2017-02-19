# Task Tree Search

Task Tree Search(TTS) is a project for creating task and motion planning algorithms that use machine learning to solve challenging problems in a variety of domains. This code provides a testbed for complex task and motion planning search algorithms. The goal is to describe example problems where actor must move around in the world and plan complex interactions with other actors or the environment that correspond to high-level symbolic states.

To run TTS examples, you will need TensorFlow and Keras, plus a number of Python packages. Note that while this package is distributed as a ROS package right now, that functionality is all missing at the moment.

For some more information on the structure of the TTS package, check out the [design overview](design.md).

## Installation

TTS can be installed either as a ROS catkin package or as an independent python package. Most features will work just fine if it is used without ROS.

  - To install TTS as a ROS package, just `git clone` it into your catkin workspace, build, re-source, and start running scripts.
  - To install TTS as an independent python package, use the `setup.py` file in the `python` directory.

To install the python packages on which TTS depends:
```
pip install h5py Theano pygame sympy matplotlib pygame gmr networkx
```

Other Required Libraries:
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras 1.1.2](https://github.com/fchollet/keras)
  - [Keras-RL](https://github.com/matthiasplappert/keras-rl/) -- it may be useful to look at [my fork](https://github.com/cpaxton/keras-rl) if you run into any issues.
  - [OpenAI Gym](https://github.com/openai/gym) -- note that you can install from `pip` as well, TTS defines its own gym environments.

Included executables (for now you need to add these to your path if you want to use LTL):
  - `ltl2ba` tool from [here](http://www.lsv.ens-cachan.fr/~gastin/ltl2ba/download.php)
  - `ltl2dstar` tool from [here](http://www.ltl2dstar.de/)

**[For developers]** Libraries referenced, but not needed as prerequisites:
  - [Tensorflow-Reinforce](https://github.com/yukezhu/tensorflow-reinforce)
  - [Guided Policy Search](https://github.com/cbfinn/gps) - `traj_opt` directly included
  - [KeRLym](https://github.com/osh/kerlym) - referenced during `Trainer` implementation

## Getting Started With Learning and Planning

All of the important scripts are in `scripts/gym`. These are based off of the Road World problem domain and associated OpenAI Gym environments, as described further on in this document.

Individual tasks and sub-tasks are referred to as "options," as per the [design overview](design.md). These are each associated with their own specific set of rules for generating environments for training, and with a set of execution constraints.

The full set of options is:
  - Default: proceed down the road.
  - Left: move to the left lane. Do not go off the road, or stay in the current lane too long.
  - Right: move to the right lane. Do not go off the road, or stay in the current lane too long.
  - Wait: sit at an intersection until it is your turn to enter it. Pass through the intersection.
  - Follow: move along behind another car. Do not leave your lane. Do not hit anything or leave the road.
  - Stop: come to a stop in a stop region.

These may remain unused:
  - Accelerate: get up to a certain speed.
  - Cruise: proceed down the road for some time.

There is a script named after each of these examples in the `scripts/gym` folder that will load and run a short version of its associated sub-problem.
To run the learning and planning code:
  - Train all models with `./scripts/gym/train.py` or `rosrun costar_task_search train.py`
  - Visualize a single planning problem with `./scripts/gym/mcts_nn_test.py` or `rosrun costar_task_search mcts_nn_test.py`
  - Use `./scripts/gym/option.py` or `rosrun costar_task_search option.py` to learn or visualize a particular road world sub-problem.
  - Use `./scripts/gym/mcts.py` or `rosrun costar_task_search mcts.py` to look at planning results.

### Training an Option

Use the script `scripts/gym/option.py`. 
```
usage: option [-h] [--agent {ddpg,lstm_ddpg,cdqn}] [--seed SEED] [--demo]
              [--visualize] [--test_episodes TEST_EPISODES] [--test_only]
              {Accelerate,Stop,Cruise,Left,Right,Wait,Follow,Finish,Default,Planning}
```

Positional arguments specify the option you want to train:
```
  {Accelerate,Stop,Cruise,Left,Right,Wait,Follow,Finish,Default,Planning}
```

### Testing MCTS

Use the script `scripts/gym/mcts.py`. 

```
usage: mcts [-h] [--option_name OPTION_NAME] [--action_set {default,nn}]
            [--rollout_type {none,simulation,action_value}]
            [--rollout_model {none,supervised,rl}] [--agent_name {none,dqn}]
            [--initialization_type {none,policy,learned_policy}]
            [--initialization_prior {default,nn}] [--verbose] [--seed SEED]
```

Run an MCTS test. Example configurations may include:

```
	mcts --rollout_type action_value --rollout_model supervised --action_set nn
	mcts --rollout_type simulation --rollout_model rl --action_set nn
	mcts --rollout_type simulation --action_set default
```

For example, you might want to compare speed for CPU vs. GPU, neural net policies vs. manual policies, etc. If so, these are your commands:
```
./mcts.py  --seed 5 --action_set nn --agent dqn --profile --cpu
./mcts.py  --seed 5 --action_set nn --agent dqn --profile
./mcts.py  --seed 5 --action_set nn --agent none --profile --cpu
./mcts.py  --seed 5 --action_set nn --agent none --profile
./mcts.py  --seed 5 --action_set default --agent dqn --profile --cpu
./mcts.py  --seed 5 --action_set default --agent dqn --profile
./mcts.py  --seed 5 --action_set default --agent none --profile --cpu
./mcts.py  --seed 5 --action_set default --agent none --profile
```

Optional arguments for the MCTS tool:
```
  -h, --help            show help message and exit
  --option_name OPTION_NAME
  --action_set {default,nn}
                        Use a set of hand-coded policies ('default') or neural
                        net options ('nn').
  --rollout_type {none,simulation,action_value}
  --rollout_model {none,supervised,rl}
  --agent_name {none,dqn}
                        Use weights learned by a particular agent (e.g,
                        'dqn'), or no weights at all ('none')
  --initialization_type {none,policy,learned_policy}
  --initialization_prior {default,nn}
  --verbose             Print out causes of termination conditions and other
                        information.
  --visualize           Display graphs and other stuff.
  --seed SEED           Random seed to use.
  --profile             Run a profiler on the planner.
  --cpu                 Force use of cpu0
  --graph               Show output graphs displaying information about the
                        chosen trajectory.
  --iter ITER           Number of MCTS iterations to perform during each step.
```

Specify the parameters you want for your particular test.

## Problem Domains

  - Grid World: navigate a busy road in a discrete grid task.
  - Road World: navigate a busy road in continuous space.
  - Needle Master: steer a needle through a sequence of gates while avoiding obstacles.
  - Robotics: mimic an expert task performance in a new environment.

### Grid World

[![Grid World](https://img.youtube.com/vi/LLs1OIIIQnw/0.jpg)](https://youtu.be/LLs1OIIIQnw)

Grid world is an ultra-simple driving task. The actor must move around a grid and navigate an intersection. Note that as the first domain implemented, this does not fully support the TTS API.

#### Run

```
create_training_data.py
learn_simple.py
```

### Road World

[![Road World](https://img.youtube.com/vi/KnNJhu8ULmc/0.jpg)](https://youtu.be/KnNJhu8ULmc)

Road world contains multiple car actors moving in continuous space. It is the first world that was developed to use the new TTS API, including all the abstract classes we have lying around. The high level Road World problem consists of three steps:
  - **proceed down route:** go to speed limit if possible, do not hit obstacles
  - **stop at stop sign:** decrease speed to zero in stop region
  - **navigate intersection:** travel through the intersection only once it is clear

In particular, we want to choose a set of high-level options from:
```
  Options = {
                Default, # D
                Accelerate, # A
                Cruise, # C
                Wait, # W
                Stop, # S
                Change Lanes Left, # L
                Change Lanes Right, # R
                Follow Car, # F
                Pass Car, # P
            }
```

Where a sample sequence on a road with a stop sign might look like:
```
  seq = [A,C,C,S,W,W,A,C,C,C]
```

Road World also has an associated "sampler" task. In this task, our goal is to choose which branch of a tree to explore next from a particular state.

### Needle Master

[![Needle Master Gameplay](https://img.youtube.com/vi/GgIznhbk-5g/0.jpg)](https://youtu.be/GgIznhbk-5g)

Example from a simple Android game. This comes with a dataset that can be used; the goal is to generate task and motion plans that align with the expert training data.

Note that this is a "driving" task, just like Road World, and they actually have a lot of similarities. The primary difference is that Needle Master, as a "planning" domain, operates over motion primitives by default.

One sub-task from the Needle Master domain is trajectory optimization. The goal is to generate an optimal trajectory in the shortest possible amount of time.

### Robotics

These examples are designed to work with ROS and a simulation of the Universal Robots UR5, KUKA LBR iiwa, or other robot. Currently in the planning stages and not developed yet.

## Contact

This code is maintained by Chris Paxton (cpaxton@jhu.edu).

## Known Issues

TTS is currently under development. This section will document missing features, bugs, or short-term implementation decisions that may influence the behavior of this code and its examples.

Missing components and features:

  - NeedleMaster conditions are wrong
  - NeedleMaster needs predicates to be implemented
  - NeedleMaster cost function is missing
  - Specify options `{Approach, Pass-Through, Connect, Exit}` for Needle Master.
  - Robotics examples with UR5 simulation.
  - Missing/incomplete trainers:
    - Implementation of REINFORCE for recurrent nets
    - Implementation of REINFORCE for continuous action spaces
    - Cross-Entropy Method has some bugs still
    - Guided Policy Search
  - Robotics examples:
    - Possibly fit neural nets or other models for these policies
    - JointStateListener needs to prune unused joint states
    - GMM policy can be fit based on data
    - Feature function based on relative position and orientation to objects
    - Demonstration reward function
  - MCTS:
    - continuous RAVE
    - kernel regression
    - double progressive widening

Other minor issues:

  - `scanEvents()` and more complex RoadWorld behavior is only supported in the lateral version of the environment for now.
  - Manual policy for lateral-motion road actors always assumes that there is only one intersection and all stops just relate to that intersection. The way to fix this is just to set actors' next events to stop+intersection_id or intersection+intersection_id rather than just 'stop' or 'intersection'.
  - `scanEvents()` assumes there is only zero or one of each of horizontal and vertical routes
  - HAS-predicate computation is currently really stupid; this means we cannot use them as failure conditions. Instead, all conditions should just be predicates, and we should look up these predicates to test conditions.
  - Meta predicate checking could be done more efficiently. It would be best if these were looked up from the set of predicates rather than holding their own copies of every predicate.
  - General speed concerns: feature computation could be faster, as could the update of the world.


