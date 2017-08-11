#!/usr/bin/env bash
rosrun costar_bullet start --robot ur5 --agent ff --features multi  \
  --model tcn_regression --si 5 -i 1000  --batch_size 64 --gui
