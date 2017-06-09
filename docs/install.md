# How to Install CoSTAR

Note: CoSTAR installation has only been tested on ROS Indigo (Ubuntu 14.04 LTS). For instructions on Indigo installation, please see [here](http://wiki.ros.org/indigo/Installation/Ubuntu). There is a prototype install script available [here](install_indigo.sh) that you can try out as well.


## Installation

TTS can be installed either as a ROS catkin package or as an independent python package. Most features will work just fine if it is used without ROS.

  - To install TTS as a ROS package, just `git clone` it into your catkin workspace, build, re-source, and start running scripts.
  - To install TTS as an independent python package, use the `setup.py` file in the `python` directory.

To install the python packages on which TTS depends:
```
pip install h5py Theano pygame sympy matplotlib pygame gmr networkx dtw pypr gym numba pyyaml
```

## Prerequisites

To use the CoSTAR plan system, you will need to install the following software packages:

  - Python (tested version 2.7.12)
  - Git (tested version 1.9.1)
  - ROS (tested ROS Indigo, Ubuntu 14.04)
  - Catkin Build Tools
  - [OpenAI Gym](https://github.com/openai/gym) -- note that you can install from `pip` as well, TTS defines its own gym environments. You do not need any of the environments, so this is best installed via `pip`.
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras 1.1.2](https://github.com/fchollet/keras)
  - [Keras-RL](https://github.com/matthiasplappert/keras-rl/) -- it may be useful to look at [my fork](https://github.com/cpaxton/keras-rl) if you run into any issues.
  - [Bullet3](https://github.com/bulletphysics/bullet3.git) -- for `costar_bullet` and for simulation examples


You can download all the required packages to use CoSTAR with ROS Indigo from the Ubuntu repositories with this command:

```
# set your ros distro 
export ROS_DISTRO=indigo

# install rosdep and catkin
sudo apt-get install -y python-catkin-pkg python-rosdep python-wstool python-catkin-tools ros-$ROS_DISTRO-catkin

# init your rosdep (if you have not already done so)
sudo rosdep init
rosdep update
```

## Step 1. Get CoSTAR Planning and Simulation Packages from Git

```
cd path/to/your/catkin_ws/src
git clone https://github.com/cburbridge/python_pcd.git
git clone gttps://github.com/jhu-lcsr/costar_objects.git
git clone https://github.com/cpaxton/lcsr_assembly.git --branch devel
git clone https://github.com/cpaxton/dmp.git --branch indigo
git clone https://github.com/cpaxton/robotiq_85_gripper.git
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
```

Note the `lcsr_assembly` package is not strictly necesary.

## Step 2. Build catkin workspace

Change directory into catkin workspace folder and run:

```
catkin build
```

