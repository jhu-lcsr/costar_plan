#!/usr/bin/env bash
rosrun costar_bullet start --robot ur5 --agent null --features multi --load \
  --model conv_lstm_regression --si 5 -i 1000  --batch_size 64 --data_file large.npz
