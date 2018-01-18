#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model pretrain_image_encoder \
  --data_file husky_data.npz \
  --lr 0.001 \
  --dropout_rate 0.2 \
  --features husky  \
  --steps_per_epoch 300 
