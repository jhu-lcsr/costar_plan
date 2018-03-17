#!/usr/bin/env bash
rosrun costar_models ctp_model_tool \
        --model secondary --data_file data.h5f \
        --lr 0.0001  --dropout_rate 0.1 \
        --features multi --batch_size 64 \
        --steps_per_epoch 500 \
        --submodel pose  --success_only $1
