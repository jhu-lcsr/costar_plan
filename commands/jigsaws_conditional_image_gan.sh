#!/usr/bin/env bash 
rosrun costar_models ctp_model_tool \
  --model conditional_image_gan \
  --data_file suturing_data.h5f \
  --batch_size 32 \
  --lr 0.0002 --dropout_rate 0.2 --features jigsaws
