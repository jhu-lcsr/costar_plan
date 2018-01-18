#!/usr/bin/env bash
rosrun costar_models ctp_model_tool --model conditional_image_gan \
  --features multi --data_file data.h5f --lr 0.0002 --dropout_rate 0.2 \
  --steps_per_epoch 500
