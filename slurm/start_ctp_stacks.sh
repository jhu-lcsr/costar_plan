#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

# models_stack: no dropouts, 2 tforms
# models_stack2: all dropouts, including on tforms
# models_stack3: no dropouts on tforms?
# models_stack4: smaller hidden arm+gripper layer and smaller hidden layers
# models_stack5: (8,8,64) and ph output
# models_stack6: (8,8,64) and noise
# sequence A:(8,8,64), options, 1 transform layer
# sequence B:(4,4,64), options, 2 transform layers
# sequence C:(4,4,64), options, 3 transform layers
# sequence D:(8,8,64), options, 3 transform layers, dense
# sequence F: dense, 128
# sequence G: dense, 32
# sequence H: dense, 32, 1 tform
# sequence I: dense, bigger
# sequence J: dense, 256, multiple tforms
# sequence L: dense, 128, using SSM now
# sequence M: try with the --sampling flag
# sequence O: dense, 128, spatial softmax, some sampling
# sequence P: as above, some fixes
# sequence Q: as above but adding success only, stripping out value + prior +
#             actor learning so that it can happen separately after we already
#             have a good world state.
# sequence R: success only, 128, spatial softmax
# sequence S: ssm, 128, include failures, use action prior
# sequence T: ssm, 128, include failures, use action prior + prev action in
#             encoder
# seqeunce U: as per T, but also use 512 weights in arm+gripper decoder
# sequence V: as per T/U, but remove prev action and remove next label
#             prediction
for lr in 0.001 #0.01
do
  # just use the adam optimizer
  for opt in adam
  do
    # what do we do about skip connections?
    for skip in 0 1
    do
      # Noise: add extra ones with no noise at all
      for noise_dim in 0 1 8 32
      do
        hd=true
        for dr in 0.01 0.125 0.25 0.5
        do
          echo "starting LR=$lr, Dropout=$dr, optimizer=$opt, use dropout in hypotheses: $hd noise=$noise_dim, skip connections = $skip"
          sbatch ctp.sh $lr $dr $opt $hd $noise_dim $skip
        done
        hd=false
        for dr in 0.125 0.25 0.5
          echo "starting LR=$lr, Dropout=$dr, optimizer=$opt, use dropout in hypotheses: $hd noise=$noise_dim, skip connections = $skip"
          sbatch ctp.sh $lr $dr $opt $hd $noise_dim $skip
        do
        done
        dr=0.0
        echo "starting LR=$lr, Dropout=$dr, optimizer=$opt, use dropout in hypotheses: $hd noise=$noise_dim, skip connections = $skip"
        sbatch ctp.sh $lr $dr $opt $hd $noise_dim $skip
      done
    done
  done
done

# ======================================
sbatch ctp_stack.sh
sbatch ctp_stack2.sh
sbatch ctp_goal.sh

