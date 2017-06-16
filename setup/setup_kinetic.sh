#!/usr/bin/env sh

export ROS_DISTRO=kinetic
export ROS_CI_DESKTOP="`lsb_release -cs`"  # e.g. [precise|trusty|...]
export CI_SOURCE_PATH=$(pwd)
export CATKIN_OPTIONS="$CI_SOURCE_PATH/catkin.options"
export ROS_PARALLEL_JOBS='-j8 -l6'
export CATKIN_WS="$HOME/costar_ws"
export COSTAR_PLAN_DIR="$HOME/costar_ws/src/costar_plan"

echo "======================================================"
echo "PYTHON"
echo "Installing python dependencies:"
echo "Installing basics from apt-get..."
sudo apt-get -y install python-pygame python-pip
echo "Installing smaller libraries from pip..."
sudo pip install -H h5py keras sympy matplotlib pygame gmr networkx dtw pypr gym PyPNG

echo "======================================================"
echo "ROS"
sudo apt-get update -qq
sudo apt-get install -y python-catkin-pkg python-rosdep python-wstool python-catkin-tools ros-$ROS_DISTRO-catkin python-dev 
source /opt/ros/$ROS_DISTRO/setup.bash
sudo rosdep init
rosdep update

echo "======================================================"
echo "CATKIN"
echo "Create catkin workspace..."
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
source /opt/ros/$ROS_DISTRO/setup.bash
catkin init
cd $CATKIN_WS/src

git clone https://github.com/cburbridge/python_pcd.git
git clone https://github.com/jhu-lcsr/costar_objects.git
git clone https://github.com/cpaxton/dmp.git --branch indigo
#git clone https://github.com/cpaxton/costar_plan.git
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
cd $CATKIN_WS/src
catkin build
#source $CATKIN_WS/devel/setup.bash

