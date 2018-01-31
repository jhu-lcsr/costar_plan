#!/bin/bash -l
#SBATCH --job-name=secondary
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu

echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

export train_discriminator=false
export train_discriminator2=false
export train_image_encoder=false
export train_multi_encoder=false
export train_conditional_image=false
export train_conditional_sampler=false
export train_predictor=true
export train_policies=true
export learning_rate=$1
export dropout=$2
export optimizer=$3
export noise_dim=$4
export loss=$5
export model=$6

export train_multi=false
export train_husky=true

echo $0 $1 $2 $3 $4 $5 $6
echo "[STACK] Training policy $model"
export MODELDIR="$HOME/.costar/stack_$learning_rate$optimizer$dropout$noise_dim$loss"
export DATASET="ctp_dec"
mkdir $MODELDIR
touch $MODELDIR/$SLURM_JOB_ID
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
  -e 100 \
  --model secondary \
  --submodel $model \
  --data_file $HOME/work/$DATASET.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --steps_per_epoch 500 \
  --noise_dim $noise_dim \
  --loss $loss \
  --batch_size 64

export MODELDIR="$HOME/.costar/husky_$learning_rate$optimizer$dropout$noise_dim$loss"
export DATASET="husky_data"
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features husky \
  -e 100 \
  --model secondary \
  --data_file $HOME/work/$DATASET.npz \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --steps_per_epoch 500 \
  --noise_dim $noise_dim \
  --loss $loss \
  --batch_size 64


