#!/usr/bin/env sh

export CATKIN_WS = $HOME/catkin_ws
export COSTAR_PLAN_DIR = $HOME/catkin_ws/src/costar_plan

echo "Create catkin workspace..."
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS/src

echo "Download dependencies and other packages..."
./install_indigo_deps.sh

