#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for lr in 0.001 0.0001; do
  export GAN_METHOD=mae
  export EPOCHS=100
  echo "Starting LR=$lr, GAN_method=$GAN_METHOD, epochs=$epochs"
  sbatch gan.sh $lr $GAN_METHOD $epochs
done


