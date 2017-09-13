#!/usr/bin/env bash
rosrun costar_bullet start --robot ur5 --task stack1 \
	--agent null --features multi -i 50000 -e 100 \
	--model predictor --data_file ~/datasets/ctp_small.npz --load \
	--si 0 --lr 0.001 --optimizer nadam 

