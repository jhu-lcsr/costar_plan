#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model goal_discriminator \
  --data_file data.h5f \
  --lr 0.001 \
  --dropout_rate 0.2 \
  --features multi \
  --batch_size 64 \
  --steps_per_epoch 100
