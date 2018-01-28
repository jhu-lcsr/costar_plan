#!/usr/bin/env bash

for lr in 0.01 0.001 0.0002 0.0001
do
  for loss in mae # logcosh
  do
    # what do we do about skip connections?
    for skip in 0 # 1
    do
      # Noise: add extra ones with no noise at all
      for noise_dim in 0 # 1 8 32
      do
        hd=true
        for dr in 0.1 0.2 # 0. 0.3 0.4 0.5
        do
          for model in value next pose actor
          do
            echo "Creating models with $opt $loss $skip $noise_dim $dr $model"
            ctp_secondary.sh $lr $dr $opt $noise_dim $loss $model
            #sbatch ctp_secondary.sh $lr $dr $opt $noise_dim $loss $model
          done
        done
      done
    done
  done
done

