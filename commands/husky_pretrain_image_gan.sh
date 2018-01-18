#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model pretrain_image_gan \
  --features husky \
  --data_file husky_data.npz \
  --lr 0.0002 --dropout_rate 0.2 --steps_per_epoch 500
