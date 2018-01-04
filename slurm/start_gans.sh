#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for lr in 0.001 0.0001; do
  export gan_method=mae
  export epochs=100
  echo "Starting LR=$lr, GAN_method=$gan_method, epochs=$epochs"
  sbatch gan.sh $lr $gan_method $epochs
done


