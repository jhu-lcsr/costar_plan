#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for lr in 0.01 0.001
do
	for opt in adam
	do
		for dr in 0.1 0.2 0.5
		do
			hd=true
			echo "starting LR=$lr, Dropout=$dr, optimizer = $opt, use dropout in hypotheses = $hd"
			sbatch ctp.sh $lr $dr $opt $hd
			hd=false
			echo "starting LR=$lr, Dropout=$dr, optimizer = $opt, use dropout in hypotheses = $hd"
			sbatch ctp.sh $lr $dr $opt $hd
		done
	done
done

# ======================================
sbatch ctp_stack.sh
sbatch ctp_stack2.sh

