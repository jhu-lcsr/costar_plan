#!/usr/bin/env sh

export ROS_DISTRO=indigo
sudo pip install h5py keras Theano pygame sympy matplotlib pygame gmr networkx dtw pypr gym

roscd
cd ../src

git clone https://github.com/cburbridge/python_pcd.git
git clone gttps://github.com/jhu-lcsr/costar_objects.git
git clone https://github.com/cpaxton/lcsr_assembly.git --branch devel
git clone https://github.com/cpaxton/dmp.git --branch indigo
git clone https://github.com/cpaxton/robotiq_85_gripper.git
rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO

