#!/usr/bin/env bash

rosrun costar_models ctp_model_tool \
  --data_file rpy.npz \
  --model pretrain_image_encoder \
  -e 1000 \
  --features multi \
  --batch_size 48 \
  --optimizer adam \
  --lr 0.001 \
  --upsampling conv_transpose \
  --use_noise true \
  --noise_dim 32 \
  --steps_per_epoch 300 \
  --dropout_rate 0.1 \
  --skip_connections 1 \
  --hypothesis_dropout 0 \
  --decoder_dropout_rate 0.1

# This is an option but is not part of the default command
#--load_model \

