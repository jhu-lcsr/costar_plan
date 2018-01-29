#!/bin/bash -l
#SBATCH --job-name=gan
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end


echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

export train_image_encoder=false

export dataset=$1
export features=$2
export learning_rate=$3
export dropout=$4
export optimizer=$5
export noise_dim=$6
export loss=$7
export wass=$8 # 'wass
export MODELDIR="$HOME/.costar/${dataset}_${learning_rate}_${optimizer}_${dropout}_${noise_dim}_${loss}_${wass}"

wass_cmd=''
if [[ $wass == wass* ]]; then wass_cmd='--use_wasserstein'; fi
	

if $train_image_encoder
then
  echo "Training encoder 1"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $HOME/work/$dataset.h5f \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --use_noise false \
    --steps_per_epoch 500 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi

echo "Training encoder gan"
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features $features \
  -e 100 \
  --model pretrain_image_gan \
  --data_file $HOME/work/$dataset.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise false \
  --steps_per_epoch 500 \
  --noise_dim $noise_dim \
  --loss $loss \
  --gan_method gan \
  --batch_size 64 \
	$wass_cmd

echo "Training conditional gan"
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features $features \
  -e 100 \
  --model conditional_image_gan \
  --data_file $HOME/work/$dataset.h5f \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --use_noise false \
  --steps_per_epoch 500 \
  --noise_dim $noise_dim \
  --loss $loss \
  --gan_method gan \
  --batch_size 64 \
	$wass_cmd

