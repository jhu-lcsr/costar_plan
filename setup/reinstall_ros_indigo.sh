#!/usr/bin/env bash

export ROS_DISTRO=indigo

sudo apt-get install -y python-catkin-pkg python-rosdep python-wstool python-catkin-tools ros-$ROS_DISTRO-catkin ros-$ROS_DISTRO-ros-base
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
