#!/bin/bash -l
#SBATCH --job-name=gan
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
export train_image_encoder=false
export learning_rate=$1
export dropout=$2
export optimizer=$3
export noise_dim=$4
export loss=$5
export MODELDIR="$HOME/.costar/stack_$learning_rate$optimizer$dropout$noise_dim$loss"

if $train_image_encoder
then
  echo "Training encoder 1"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi

echo "Training encoder gan"
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model pretrain_image_gan \
  --data_file $HOME/work/$DATASET.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $noise_dim \
  --loss $loss \
  --gan_method gan \
  --batch_size 64
