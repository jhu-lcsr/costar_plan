#!/usr/bin/env bash
echo "format: $0 [option_num] [optional_arg]"
rosrun costar_models ctp_model_tool \
  --model policy \
  --data_file husky_data.npz \
  --lr 0.01 \
  --dropout_rate 0.2 \
  --features husky \
  --batch_size 32  \
  --steps_per_epoch 500 \
  --option_num $1 $2
