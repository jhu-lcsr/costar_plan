#!/bin/bash -l
#SBATCH --job-name=ctpZ
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu


echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

export DATASET="ctp_dec"
export learning_rate=$1
export dropout=$2
export optimizer=$3
export noise_dim=$4
export loss=$5
export train_policies=true
export MODELDIR="$HOME/.costar/stack_$learning_rate$optimizer$dropout$noise_dim$loss"

if $train_policies
then
  for opt in $(seq 0 36)
  do
    $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
      --features multi \
      -e 100 \
      --model policy \
      --data_file $HOME/work/$DATASET.h5f \
      --lr $learning_rate \
      --dropout_rate $dropout \
      --model_directory $MODELDIR/ \
      --optimizer $optimizer \
      --use_noise true \
      --steps_per_epoch 500 \
      --loss $loss \
      --option_num $opt \
      --skip_connections 1 \
      --batch_size 64 # --retrain 
      #--success_only \
    done
fi
