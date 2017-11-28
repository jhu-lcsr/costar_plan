# How to Install CoSTAR

Note: CoSTAR installation has only been tested on ROS Indigo (Ubuntu 14.04 LTS). For instructions on Indigo installation, please see [here](http://wiki.ros.org/indigo/Installation/Ubuntu). There is a prototype install script available [here](install_indigo.sh) that you can try out as well.


## Installation

We provide two setup scripts to create most of your workspace:
  - [Setup script for ROS Indigo/Ubuntu 14.04](../setup/setup_indigo.sh)
  - [Setup script for ROS Kinetic/Ubuntu 16.04](../setup/setup_kinetic.sh)

**NOTE THAT THESE SCRIPTS ARE THE PREFERRED WAY TO INSTALL THIS PACKAGE.** They are extensively tested on travis and you should not see any problems.

The second option is [Docker](https://hub.docker.com/r/alee156/costar/). This should create a container with a version of the CTP environment.

If none of these options work on your platform for whatever reason, you can always try a manual install.

## Docker Install

Still in development. Simply run:
```
sudo docker run -ti alee156/costar 
```
to get started.

## Manual Install

The following are manual installation instructions for the CTP package and simulation. They are not guaranteed to be up to date; follow at your own risk.

### Prerequisites

TTS can be installed either as a ROS catkin package or as an independent python package. Most features will work just fine if it is used without ROS.

  - To install TTS as a ROS package, just `git clone` it into your catkin workspace, build, re-source, and start running scripts.
  - To install TTS as an independent python package, use the `setup.py` file in the `python` directory.

To install the python packages on which TTS depends:
```
pip install h5py Theano pygame sympy matplotlib pygame gmr networkx dtw pypr gym numba pyyaml PyPNG
```

To use the CoSTAR plan system, you will need to install the following software packages:

  - Python (tested version 2.7.12)
  - Git (tested version 1.9.1)
  - ROS (tested ROS Indigo, Ubuntu 14.04)
  - Catkin Build Tools
  - [OpenAI Gym](https://github.com/openai/gym)
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras 2](https://github.com/fchollet/keras)
  - [Keras-RL](https://github.com/matthiasplappert/keras-rl/)
  - [Bullet3](https://github.com/bulletphysics/bullet3.git)


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

#### Nvidia CudaNN

GPU setup for tensorflow is fairly easy, and you can find the instructions on the website. If it causes problems:

```
sudo pip uninstall tensorflow && sudo pip install tensorflow-gpu --upgrade
```

Then download the appropriate CudaNN libraries. For Ubuntu 14.04 and Tensforflow 1.4, these are easily installed with:
```
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb 
sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb 
```

### Step 1. Get CoSTAR Planning and Simulation Packages from Git

After this, you can download our various prerequisites from git. These packages are not qutonamtically installed from the package manager.

```
cd path/to/your/catkin_ws/src
# Contains larger binary files -- meshes, etc.
git clone gttps://github.com/jhu-lcsr/costar_objects.git

# Utility associated with costar_objects repo
git clone https://github.com/cburbridge/python_pcd.git

# Simple DMP library
git clone https://github.com/cpaxton/dmp.git --branch indigo

# Robotiq 85 gripper library
git clone https://github.com/cpaxton/robotiq_85_gripper.git

# Used to install ROS dependences from archive.
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
```

### Step 2. Build catkin workspace

Change directory into catkin workspace folder and run:

```
catkin build
```

