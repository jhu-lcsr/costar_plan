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
export learning_rate=0.001
export dropout=0.01
export optimizer=adam
export noise_dim=0
export loss=mae
export model=actor

export train_multi=false
export train_husky=true

echo "[STACK] Training policy $model"
export MODELDIR="$HOME/.costar/stack_$learning_rate$optimizer$dropout$noise_dim$loss"
export DATASET="ctp_dec"
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
