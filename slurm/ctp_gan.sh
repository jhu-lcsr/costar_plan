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

OPTS=$(getopt -o d:f: --long dataset:,features:,lr:,dr:,opt:,noisedim:,loss:,wass,no-wass,noise,retrain,train-img-encoder,train-gan-encoder -n ctp_gan -- "$@")

train_image_encoder=false
train_gan_image_encoder=false
dataset=''
features=''
lr=0.001
dropout=0.1
optimizer=adam
noise_dim=1
loss=mae
wass=false
use_noise=false
retrain=false

while true; do
  case "$1" in
    -d | --dataset) dataset="$2"; shift 2 ;;
    -f | --features) features="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --dr) dropout="$2"; shift 2 ;;
    --opt) optimizer="$2"; shift 2 ;;
    --noisedim) noise_dim="$2"; shift 2 ;;
    --noise) use_noise=true; shift ;;
    --loss) loss="$2"; shift 2 ;;
    --wass) wass=true; shift ;;
    --no-wass) wass=false; shift ;;
    --retrain) retrain=true; shift ;;
    --train-img-encoder) train_image_encoder=true; shift ;;
    --train-gan-encoder) train_gan_encoder=true; shift ;;
    --) shift; break ;;
    *) echo "Internal error!" ; exit 1 ;;
  esac
done

[[ $dataset != '' ]] && echo 'Dataset is mandatory!' && exit 1
[[ $features != '' ]] && echo 'Features are mandatory!' && exit 1


MODELDIR="$HOME/.costar/${dataset}_${lr}_${optimizer}_${dropout}_${noise_dim}_${loss}_${wass}_${use_noise}"

touch $MODELDIR/$SLURM_JOB_ID

# Handle different Marcc layouts
data_dir=$HOME/work/$dataset
[[ ! -d $data_dir ]] && data_dir=$HOME/work/dev_yb/$dataset

data_suffix=h5f
[[ $features == husky ]] && data_suffix=npz

data_dir=${data_dir}.${data_suffix}

wass_cmd=''
$wass && wass_cmd='--wasserstein'

use_noise_cmd=''
$use_noise && use_noise_cmd='--use_noise'

retrain_cmd=''
$retrain && retrain_cmd='--retrain'

if $train_image_encoder; then
  echo "Training discriminator"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model discriminator \
    --data_file $data_dir \
    --lr $lr \
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
    --lr $lr \
    --dropout_rate $dropout \
    --model_directory $MODELDIR/ \
    --optimizer $optimizer \
    --steps_per_epoch 300 \
    --noise_dim $noise_dim \
    --loss $loss \
    --batch_size 64
fi
if $train_gan_image_encoder; then
  echo "Training encoder gan"
  $HOME/costar_plan/costar_models/scripts/ctp_model_tool \
    --features $features \
    -e 100 \
    --model pretrain_image_gan \
    --data_file $data_dir \
    --lr $lr \
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
  --lr $lr \
  --dropout_rate $dropout \
  --model_directory $MODELDIR/ \
  --optimizer $optimizer \
  --steps_per_epoch 100 \
  --noise_dim $noise_dim \
  --loss $loss \
  --gan_method gan \
  --batch_size 64 \
  $wass_cmd \
  $use_noise_cmd

