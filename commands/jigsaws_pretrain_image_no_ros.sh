#!/usr/bin/env bash
CTP_MODELS=$HOME/costar_plan/costar_models
$CTP_MODELS/scripts/ctp_model_tool \
  --model pretrain_image_encoder \
  --data_file suturing_data.h5f \
  --lr 0.001 \
  --dropout_rate 0.2 \
  --features jigsaws \
  --batch_size 32 $1
