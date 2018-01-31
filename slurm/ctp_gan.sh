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

export train_image_encoder=true
export train_gan_image_encoder=false

export dataset=$1
export features=$2
export learning_rate=$3
export dropout=$4
export optimizer=$5
export noise_dim=$6
export loss=$7
export wass=$8 # 'wass
export use_noise=$9
export MODELDIR="$HOME/.costar/${dataset}_${learning_rate}_${optimizer}_${dropout}_${noise_dim}_${loss}_${wass}_${use_noise}"

# Handle different Marcc layouts
data_dir=$HOME/work/$dataset
if [[ ! -d $data_dir ]]; then
  data_dir=$HOME/work/dev_yb/$dataset
fi
if [[ $features == husky ]]; then
  data_dir=${data_dir}.npz
else
  data_dir=${data_dir}.h5f
fi

wass_cmd=''
if [[ $wass == wass* ]]; then wass_cmd='--wasserstein'; fi

use_noise_cmd=''
if [[ $use_noise == true ]]; then use_noise_cmd='--use_noise'; fi
	
if $train_image_encoder; then
  echo "Training discriminator"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model discriminator \
    --data_file $data_dir \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 300 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64

  echo "Training non-gan image encoder"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $data_dir \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 300 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi
if $train_gan_image_encoder; then
  echo "Training encoder gan: no wasserstein"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model pretrain_image_gan \
    --data_file $data_dir \
    --lr $learning_rate \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 100 \
    --noise_dim $noise_dim \
    --loss $loss \
    --gan_method gan \
    --batch_size 64 \
    $wass_cmd
fi

echo "Training conditional gan"
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features $features \
  -e 100 \
  --model conditional_image_gan \
  --data_file $data_dir \
  --lr $learning_rate \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --steps_per_epoch 100 \
  --noise_dim $noise_dim \
  --loss $loss \
  --gan_method gan \
  --batch_size 64 \
  $wass_cmd

