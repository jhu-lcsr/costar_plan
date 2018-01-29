#!/usr/bin/env bash
rosrun costar_models ctp_hidden.py \
        --model conditional_image \
        --data_file husky_data.npz \
        --lr 0.001 --dropout_rate 0.2 \
        --features husky \
        --steps_per_epoch 500 
