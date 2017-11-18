#!/usr/bin/env bash

export ROS_DISTRO=indigo
export ROS_CI_DESKTOP="`lsb_release -cs`"  # e.g. [precise|trusty|...]
export CI_SOURCE_PATH=$(pwd)
export CATKIN_OPTIONS="$CI_SOURCE_PATH/catkin.options"
export ROS_PARALLEL_JOBS='-j8 -l6'
export CATKIN_WS="$HOME/costar_ws"
export COSTAR_PLAN_DIR="$HOME/costar_ws/src/costar_plan"

sudo apt-get update -qq

sudo apt-get install -y python-catkin-pkg python-rosdep python-wstool \
  python-catkin-tools ros-$ROS_DISTRO-catkin ros-$ROS_DISTRO-ros-base
source /opt/ros/$ROS_DISTRO/setup.bash
sudo rosdep init
rosdep update

cd $CATKIN_WS
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
