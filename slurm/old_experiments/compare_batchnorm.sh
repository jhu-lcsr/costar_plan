#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

sbatch ff.sh 0.001 1
sbatch ff.sh 0.001 0
sbatch ff.sh 0.0001 0
sbatch ff.sh 0.0001 1

