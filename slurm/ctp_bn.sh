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
export MODELDIR="$HOME/.costar/stack_bn"

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
    --use_noise \
    --steps_per_epoch 500 \
    --noise_dim $noise_dim \
    --loss $loss \
    --use_batchnorm 0 \
    --batch_size 64
fi

if $train_multi_encoder
then
  echo "Training encoder 2"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_sampler \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --use_noise \
    --steps_per_epoch 500 \
    --noise_dim $noise_dim \
    --loss $loss \
    --use_batchnorm 0 \
    --batch_size 64
fi

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_image \
  --data_file $HOME/work/$DATASET.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise \
  --steps_per_epoch 500 \
  --loss $loss \
  --use_batchnorm 0 \
  --batch_size 64

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_sampler2 \
  --data_file $HOME/work/$DATASET.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise \
  --steps_per_epoch 500 \
  --loss $loss \
  --batch_size 64

if $train_predictor
then
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features multi \
    -e 100 \
    --model predictor \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --use_noise \
    --steps_per_epoch 500 \
    --loss $loss \
    --skip_connections 1 \
    --batch_size 64 # --retrain 
    #--success_only \
fi

