#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model discriminator \
  --data_file suturing_data2.h5f \
  --lr 0.001 \
  --dropout_rate 0.2 \
  --features jigsaws \
  --batch_size 64 \
  --steps_per_epoch 500 

