#!/usr/bin/env bash

rosrun costar_bullet start \
  --robot ur5 --task stack1 --agent task \
  -i 1 --features multi  --verbose \
  --seed 9 --success_only  --cpu \
  --save --data_file one.npz \
  --collection_mode next --collect_trajectories
