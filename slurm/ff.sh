#!/bin/bash -l
#SBATCH --job-name=batchnorm_test
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu


export DATASET="ctp_dec"
export CMD=$HOME/costar_plan/costar_models/scripts/ctp_model_tool
export MODELDIR="$HOME/.costar/stack_BN$1$2"
export epochs=200
export lr=$1
export batchnorm=$2

$CMD \
  --data_file $HOME/work/$DATASET.h5f \
  --model_directory $MODELDIR/ \
  --model hierarchical \
  -e $epochs \
  --features multi \
  --batch_size 64  \
  --optimizer adam \
  --lr $lr \
  --upsampling conv_transpose \
  --steps_per_epoch 500 \
  --dropout_rate 0.2 \
  --loss mae \
  --use_batchnorm $batchnorm

