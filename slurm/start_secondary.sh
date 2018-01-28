#!/usr/bin/env bash

noise_dim=0
hd=true
opt=adam
for lr in 0.01 0.001 0.0002 0.0001
do
  for loss in mae # logcosh
  do
    # what do we do about skip connections?
    for skip in 0 # 1
    do
      for dr in 0.1 0.2 # 0. 0.3 0.4 0.5
      do
        for model in value next pose actor q
        do
          echo "Creating models with $opt $loss $skip $noise_dim $dr $model"
          sbatch ctp_secondary.sh $lr $dr $opt $noise_dim $loss $model
        done
      done
    done
  done
done

