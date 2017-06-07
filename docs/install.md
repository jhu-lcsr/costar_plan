# How to Install CoSTAR

Note: CoSTAR installation has only been tested on ROS Indigo (Ubuntu 14.04 LTS). For instructions on Indigo installation, please see [here](http://wiki.ros.org/indigo/Installation/Ubuntu). There is a prototype install script available [here](install_indigo.sh) that you can try out as well.


## Installation

TTS can be installed either as a ROS catkin package or as an independent python package. Most features will work just fine if it is used without ROS.

  - To install TTS as a ROS package, just `git clone` it into your catkin workspace, build, re-source, and start running scripts.
  - To install TTS as an independent python package, use the `setup.py` file in the `python` directory.

To install the python packages on which TTS depends:
```
pip install h5py Theano pygame sympy matplotlib pygame gmr networkx dtw pypr gym numba
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

This is listed separately for now.
```
cd path/to/your/catkin_ws/src
git clone https://github.com/cburbridge/python_pcd.git
git clone gttps://github.com/jhu-lcsr/costar_objects.git
git clone https://github.com/cpaxton/lcsr_assembly.git --branch devel
git clone https://github.com/cpaxton/dmp.git --branch indigo
git clone https://github.com/cpaxton/robotiq_85_gripper.git
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
```

## Step 2. Build catkin workspace

Change directory into catkin workspace folder and run:

```
catkin build
```
 
Note: Please use this command to build your catkin workspace instead of `catkin_make`.

***Debugging:***

CoSTAR is distributed as a single large package. This means that 

* For problems with message: "Assertion failed: check for file existence, but filename (RT_LIBRARY-NOTFOUND) unset.  Message: RT Library." Please clean the catkin build folder and rebuild.  
* For problems in relation to predicator_collision, please use the following command:  
`cd path/to/costar_stack/costar_predicator/predicator_collision`  
`touch CATKIN_IGNORE`
3. For problems with message: "Errors: iiwa_hw:make". Please use the following command:
`cd path/to/iiwa_stack/iiwa_hw`  
`touch CATKIN_IGNORE`
4. For problems with message: "[ERROR] [1474482887.121864954, 0.669000000]: Initializing controller 'joint_state_controller' failed". Please try installing the following packages:  
`sudo apt-get install ros-indigo-joint-state-controller`


## Step 3. Run simulation
[Optional] Checkout an example CoSTAR workspace from github into ~/.costar by running:

```
cd && git clone git@github.com:cpaxton/costar_files.git .costar\
```

Now you can run the simulation with following commands. Please remember to run `source ~/catkin_ws/devel/setup.bash` before executing any of these commands, and consider adding this line to ~/.bashrc.

```
roslaunch iiwa_gazebo iiwa_gazebo.launch trajectory:=false  
roslaunch costar_bringup iiwa14_s_model.launch sim:=true start_sim:=false  
```


*If everything shows up, CoSTAR system is then successfully installed. Enjoy!*

The top should say "Robot Mode: Idle." If you installed the sample workspace, open the Menu (lower right) and click Waypoints. Put the robot into Servo mode, highlight some waypoints, and click Servo to Waypoint (the purple button on the Waypoints popup). Not all the waypoints are guaranteed to work for this robot, but you should be able to get the robot to move.

CoSTAR is currently set up to launch our two testbed systems: a KUKA LBR iiwa 14 with a 3-finger Robotiq gripper and a Universal Robots UR5 with a 2-finger Robotiq gripper. We plan to add some funcitonality to support additional platforms.

If you are interested in supporting another platform or run into other issues trying to run this code, please contact Chris Paxton (cpaxton@jhu.edu).

