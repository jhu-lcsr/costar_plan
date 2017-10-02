#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for lr in [0.1, 0.01, 0.001]
do
	for dr in [0.1, 0.2, 0.5]
	do
		echo "starting LR=$lr, Dropout=$dr"
		sbatch ctp.sh $lr $dr
	done
done

# ======================================
#sbatch ctp_stack.sh
#sbatch ctp_stack_upsampling2d.sh
#sbatch ctp_stack2.sh
#sbatch ctp_stack_small.sh
