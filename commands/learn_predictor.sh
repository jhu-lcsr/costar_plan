#!/usr/bin/env bash
#rosrun costar_models ctp_model_tool \
#	--features multi -i 50000 -e 100 \
#	--model predictor --data_file ~/datasets/ctp_test.npz \
#	--si 0 --lr 0.001 --optimizer nadam \
#        --batch_size 64 

rosrun costar_models ctp_model_tool --data_file rpy.npz \
  --model pretrain_image_encoder \
  -e 150 \
  --features multi \
  --batch_size 64 \
  --optimizer adam \
  --lr 0.01 \
  --upsampling conv_transpose \
  --use_noise true \
  --noise_dim 32 \
  --steps_per_epoch 500 \
  --dropout_rate 0. \
  --skip_connections 0 \
  --loss mae \
  --load_model \
  --decoder_dropout_rate 0. --hypothesis_dropout 0

rosrun costar_models ctp_model_tool --data_file rpy.npz \
  --model pretrain_state_encoder \
  -e 150 \
  --features multi \
  --batch_size 128  \
  --optimizer adam \
  --lr 0.01 \
  --upsampling conv_transpose \
  --use_noise true \
  --noise_dim 32 \
  --steps_per_epoch 500 \
  --dropout_rate 0. \
  --skip_connections 0 \
  --loss mae \
  --load_model \
  --decoder_dropout_rate 0. --hypothesis_dropout 0



