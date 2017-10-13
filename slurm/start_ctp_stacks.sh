#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for lr in 0.001 # 0.01
do
	for opt in adam
	do
    for noise_dim in 1 32 # 8
    do
      hd=true
      for dr in 0.125 # 0.5
      do
        echo "starting LR=$lr, Dropout=$dr, optimizer=$opt, use dropout in hypotheses: $hd noise=$noise_dim"
        sbatch ctp.sh $lr $dr $opt $hd $noise_dim
      done
      hd=false
      dr=0.0
      echo "starting LR=$lr, Dropout=$dr, optimizer=$opt, use dropout in hypotheses: $hd noise=$noise_dim"
      sbatch ctp.sh $lr $dr $opt $hd $noise_dim
    done
	done
done

# ======================================
sbatch ctp_stack.sh
sbatch ctp_stack2.sh
sbatch ctp_goal.sh
