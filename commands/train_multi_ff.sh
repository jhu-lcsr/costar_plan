#!/usr/bin/env bash
rosrun costar_bullet start --robot ur5 --task blocks --agent null \
  --features multi -i 1000 --model ff_regression \
  --data_file large.npz --load

