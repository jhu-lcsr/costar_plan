#!/bin/bash -l
#SBATCH --job-name=jigsaws
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
export train_discriminator1=false
export train_discriminator2=true
export train_image_encoder=true
export learning_rate=$1
export dropout=$2
export optimizer=$3
export noise_dim=$4
export loss=$5
export retrain=$6
export use_disc=$7
#export MODELDIR="$HOME/.costar/suturing_$learning_rate$optimizer$dropout$noise_dim$loss"
export MODELROOT="$HOME/.costar"
export SUBDIR="suturing_$learning_rate$optimizer$dropout$noise_dim$loss"

retrain_cmd=""
if $retrain
then
  retrain_cmd="--retrain"
  SUBDIR=${SUBDIR}_retrain
fi

use_disc_cmd=""
if ! $use_disc ; then
  use_disc_cmd="--no_disc"
  SUBDIR=${SUBDIR}_nodisc
fi

export MODELDIR="$MODELROOT/$SUBDIR"
mkdir $MODELDIR
touch $MODELDIR/$SLURM_JOB_ID

export learning_rate_disc=0.01
export learning_rate_enc=0.01

if $train_discriminator1 && $use_disc ; then
  echo "Training discriminator 1"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    -e 100 \
    --model discriminator \
    --data_file $HOME/work/$DATASET.h5f \
    --preload \
    --features jigsaws \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 150 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi
if $train_discriminator2
then
  echo "Training discriminator 2"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    -e 100 \
    --model goal_discriminator \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $learning_rate \
    --features jigsaws \
    --preload \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 150 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi



if $train_image_encoder
then
  echo "Training encoder 1"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --features jigsaws \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --preload \
    --steps_per_epoch 300 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64 $use_disc
fi

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_image \
  --data_file $HOME/work/$DATASET.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --features jigsaws \
  --optimizer $optimizer \
  --steps_per_epoch 300 \
  --preload \
  --loss $loss \
  --batch_size 64 $retrain_cmd $use_disc_cmd

