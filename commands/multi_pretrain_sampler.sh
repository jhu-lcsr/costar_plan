#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
  --model pretrain_sampler \
  --features multi \
  --data_file data.h5f \
  --lr 0.001  \
  --dropout_rate 0.1 $1
