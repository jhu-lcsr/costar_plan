#!/usr/bin/env bash
rosrun costar_bullet start --robot ur5 --task blocks --agent ff \
  --features multi -i 1000 --model ff_regression --gui 
