# CoSTAR Plan

[![Build Status](https://travis-ci.com/cpaxton/costar_plan.svg?token=13PmLzWGjzrfxQvEyWp1&branch=master)](https://travis-ci.com/cpaxton/costar_plan)

CoSTAR Plan is for deep learning with robots divided into two main parts: The CoSTAR Task Planner (CTP) library and CoSTAR Hyper.

### CoSTAR Task Planner (CTP)

Code for the paper [Visual Robot Task Planning](https://arxiv.org/abs/1804.00062).

### [CoSTAR Hyper](costar_hyper/README.md)

Code for the paper [Training Frankenstein's Creature To Stack: HyperTree Architecture Search](https://sites.google.com/view/hypertree-renas/home).

[![Training Frankenstein's Creature To Stack: HyperTree Architecture Search](https://img.youtube.com/vi/1MV7slHnMX0/0.jpg)](https://youtu.be/1MV7slHnMX0 "Training Frankenstein's Creature To Stack: HyperTree Architecture Search")


Code instructions are in the [CoSTAR Hyper README.md](costar_hyper/README.md).

### Supported Datasets

  - [CoSTAR Block Stacking Dataset](https://sites.google.com/site/costardataset) 
[![2018-06-21-23-21-49_example000004 success_tiled](https://user-images.githubusercontent.com/55744/47169252-ff1e3380-d2d0-11e8-97ed-1d747d97ea11.jpg)](https://sites.google.com/site/costardataset "CoSTAR Block stacking Dataset")
  - [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)
  - [Google Brain Grasping Dataset](https://sites.google.com/site/brainrobotdata/home/grasping-dataset)


# CoSTAR Task Planner (CTP)


The CoSTAR Planner is part of the larger [CoSTAR project](https://github.com/cpaxton/costar_stack/). It integrates some learning from demonstration and task planning capabilities into the larger CoSTAR framework in different ways.

[![Visual Task Planning](https://img.youtube.com/vi/Rk4EDL4B7zQ/0.jpg)](https://youtu.be/Rk4EDL4B7zQ "Visual Task Planning")

Specifically it is a project for creating task and motion planning algorithms that use machine learning to solve challenging problems in a variety of domains. This code provides a testbed for complex task and motion planning search algorithms.

The goal is to describe example problems where the actor must move around in the world and plan complex interactions with other actors or the environment that correspond to high-level symbolic states. Among these is our Visual Task Planning project, in which robots learn representations of their world and use these to imagine possible futures, then use these for planning.

To run deep learning examples, you will need TensorFlow and Keras, plus a number of Python packages. To run robot experiments, you'll need a simulator (Gazebo or PyBullet), and ROS Indigo or Kinetic. Other versions of ROS may work but have not been tested. If you want to stick to the toy examples, you do not need to use this as a ROS package.

*About this repository:* CTP is a _single-repository_ project. As such, all the custom code you need should be in one place: here. There are exceptions, such as the [CoSTAR Stack](https://github.com/cpaxton/costar_stack/) for real robot execution, but these are generally not necessary. The minimal installation of CTP is just to install the `costar_models` package as a normal [python package](https://github.com/cpaxton/costar_plan/tree/master/costar_models/python) ignoring everything else.

# CTP Datasets

  - PyBullet Block Stacking [download tar.gz](https://github.com/cpaxton/costar_plan/releases/download/v0.6.0/simdata.tar.gz)
  - Sample Husky Data [download tar.gz](https://github.com/cpaxton/costar_plan/releases/download/v0.6.0/husky_data.tar.gz)
  - Classic CoSTAR Real Robot Data [download tar.gz](https://github.com/cpaxton/costar_plan/releases/download/v0.6.0/sample_real_ur5_robot_data.tar.gz)
     - Early version, deprecated in lieu of the full [CoSTAR Block Stacking Dataset](sites.google.com/site/costardataset).


# Contents
  - [0. Introduction](docs/introduction.md)
  - [1. Installation Guide](docs/install.md)
    - [1.1 Docker Instructions](docs/docker_instructions.md)
    - [1.2 Application domains](docs/domains.md)
  - [2. Approach](docs/approach.md): about CTP
    - [2.1 Software Design](docs/design.md): high-level notes
  - [3. Machine Learning Models](docs/learning.md): using the command line tool
    - [3.1 Data collection](docs/collect_data.md): data collection with a real or simulated robot
    - [3.2 MARCC instructions](docs/marcc.md): learning models using JHU's MARCC cluster
    - [3.3 Generative Adversarial Models](docs/learning_gan.md)
    - [3.4 SLURM Utilities](docs/slurm_utils.md): tools for using slurm on MARCC
  - [4. Creating and training a custom task](docs/task_learning.md): overview of task representations
    - [4.1 Generating a task from data](docs/generate_task_model.md): generate task from demonstrations
    - [4.2 Task Learning](docs/task_learning_experiments.md): specific details
  - [5. CoSTAR Simulation](docs/simulation.md): simulation intro
    - [5.1 Simulation Experiments](docs/simulation-experiments.md): information on experiment setup
    - [5.2 PyBullet Sim](docs/pybullet.md): an alternative to Gazebo that may be preferrable in some situations
    - [5.3 costar_bullet quick start](docs/costar_bullet.md): How to run tasks, generate datasets, train models, and extend costar_bullet with your own components.
    - [5.4 Adding a robot to the ROS code](docs/add_a_robot.md): NOT using Bullet sim
  - [6. Husky robot](docs/husky.md): Start the APL Husky sim
  - [7. TOM robot](docs/tom.md): introducing the TOM robot from TUM
    - [7.1 TOM Data](docs/tom_data.md): data necessary for TOM
    - [7.2 The Real TOM](docs/tom_real_robot.md): details about parts of the system for running on the real TOM
  - [8. CoSTAR Robot](docs/costar_real_robot.md): execution with a standard UR5

# Package/folder layout
  - [CoSTAR Simulation](costar_simulation/Readme.md): Gazebo simulation and ROS execution
  - [CoSTAR Task Plan](costar_task_plan/Readme.md): the high-level python planning library
  - [CoSTAR Gazebo Plugins](costar_gazebo_plugins/Readme.md): assorted plugins for integration
  - [CoSTAR Models](costar_models/Readme.md): tools for learning deep neural networks
  - [CTP Tom](ctp_tom/Readme.md): specific bringup and scenarios for the TOM robot from TU Munich
  - [CTP Visual](ctp_visual/Readme.md): visual robot task planner
  - `setup`: contains setup scripts
  - `slurm`: contains SLURM scripts for running on MARCC
  - `command`: contains scripts with example CTP command-line calls
  - `docs`: markdown files for information that is not specific to a particular ROS package but to all of CTP
  - `photos`: example images
  - `learning_planning_msgs`: ROS messages for data collection when doing learning from demonstration in ROS
  - Others are temporary packages for various projects

Many of these sections are a work in progress; if you have any questions shoot me an email (`cpaxton@jhu.edu`).

# Contact

This code is maintained by:

 - Chris Paxton (cpaxton@jhu.edu).
 - Andrew Hundt (ATHundt@gmail.com)

# Cite


[Visual Robot Task Planning](https://arxiv.org/abs/1804.00062)

[Training Frankenstein's Creature To Stack: HyperTree Architecture Search](https://sites.google.com/view/hypertree-renas/home)