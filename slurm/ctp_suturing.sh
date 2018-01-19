#!/bin/bash -l
#SBATCH --job-name=ctpHusky
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu


echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

export DATASET="suturing_data2"
export train_image_encoder=true
export learning_rate=$1
export dropout=$2
export optimizer=$3
export noise_dim=$4
export loss=$5
export MODELDIR="$HOME/.costar/suturing_$learning_rate$optimizer$dropout$noise_dim$loss"

if $train_image_encoder
then
  echo "Training encoder 1"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $HOME/work/$DATASET.npz \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --features jigsaws \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_image \
  --data_file $HOME/work/$DATASET.npz \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise true \
  --steps_per_epoch 500 \
  --loss $loss \
  --batch_size 64

