#!/bin/bash -l

set -e
set -x
set -u

# compile and install the current code
cd $HOME/costar_plan/costar_models/python
python setup.py install --user
cd -

for lr in 0.001 0.0001; do
  export gan_method=mae
  export epochs=100
  echo "Starting LR=$lr, GAN_method=$gan_method, epochs=$epochs"
  sbatch gan.sh $lr $gan_method $epochs
done


