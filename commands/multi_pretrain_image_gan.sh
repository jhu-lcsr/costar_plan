#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model pretrain_image_gan \
  --features multi \
  --data_file data.h5f \
  --lr 0.0002  \
  --steps_per_epoch 300 \
  --dropout_rate 0.2 $1
