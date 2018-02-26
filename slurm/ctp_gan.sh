#!/bin/bash -l
#SBATCH --job-name=gan
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end

# Check if running on marcc
if [[ -z ${SLURM_JOB_ID+x} ]]; then marcc=false; else marcc=true; fi
if $marcc; then
  echo "Running $@ on $SLURMD_NODENAME ..."
  module load tensorflow/cuda-8.0/r1.3
else
  echo "Not running on Marcc"
fi

## Option Processing ----
OPTS=$(getopt -o '' --long lr:,dr:,opt:,noisedim:,loss:,wass,no_wass,noise,retrain,gan_encoder,skip_encoder,load_model,suffix:,multi,husky,jigsaws,no_resume,epochs1:,epochs2:,enc_dir:,skip_cond -n ctp_gan -- "$@")

[[ $? != 0 ]] && echo "Failed parsing options." && exit 1

gan_encoder=false
skip_encoder=false
lr=0.001
dropout=0.1
optimizer=adam
noise_dim=1
loss=mae
wass=false
use_noise=false
retrain=false
load_model=false
dataset=''
features=''
suffix=''
resume=true # resume a job
epochs1=100
epochs2=100
enc_dir=''
skip_cond=false

echo "$OPTS"
eval set -- "$OPTS"

while true; do
  case "$1" in
    --lr) lr="$2"; shift 2 ;;
    --dr) dropout="$2"; shift 2 ;;
    --opt) optimizer="$2"; shift 2 ;;
    --noisedim) noise_dim="$2"; shift 2 ;;
    --noise) use_noise=true; shift ;;
    --loss) loss="$2"; shift 2 ;;
    --wass) wass=true; shift ;;
    --no_wass) wass=false; shift ;;
    --retrain) retrain=true; shift ;;
    --skip_encoder) skip_encoder=true; shift ;;
    --skip_cond) skip_cond=true; shift ;;
    --gan_encoder) gan_encoder=true; shift ;;
    --load_model) load_model=true; shift ;;
    --multi) dataset=ctp_dec; features=multi; shift ;;
    --husky) dataset=husky_data; features=husky; shift ;;
    --jigsaws) dataset=suturing_data2; features=jigsaws; shift ;;
    --suffix) suffix="$2"; shift 2 ;;
    --no_resume) resume=false; shift ;;
    --epochs1) epochs1="$2"; shift 2 ;;
    --epochs2) epochs2="$2"; shift 2 ;;
    --enc_dir) enc_dir="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Internal error!" ; exit 1 ;;
  esac
done

# positional arguments
[[ ! -z "$1" ]] && dataset="$1"
[[ ! -z "$2" ]] && features="$2"

[[ $dataset == '' ]] && echo 'Dataset is mandatory!' && exit 1
[[ $features == '' ]] && echo 'Features are mandatory!' && exit 1

## End of option processing ---------------------

if $wass; then wass_dir=wass; else wass_dir=nowass; fi
if $use_noise; then noise_dir=noise; else noise_dir=nonoise; fi
if $retrain; then retrain_dir=retrain; else retrain_dir=noretrain; fi
if $gan_encoder; then gan_dir=ganenc; else gan_dir=noganenc; fi

# Handle model directory
MODELDIR="$HOME/.costar/${dataset}_${lr}_${optimizer}_${dropout}_${noise_dim}_${loss}_${wass_dir}_${noise_dir}_${gan_dir}_${retrain_dir}${suffix}"

[[ ! -d $MODELDIR ]] && mkdir -p $MODELDIR

# Get rid of old status files if not resuming
! $resume && rm "$MODELDIR"/status*.txt

# Set data directory
if $marcc; then
  touch $MODELDIR/$SLURM_JOB_ID
  # Handle different Marcc layouts
  data_dir=$HOME/work/$dataset
  [[ ! -d $data_dir ]] && data_dir=$HOME/work/dev_yb/$dataset
else
  data_dir=$dataset
fi
if [[ $features == husky ]]; then data_suffix=npz; else data_suffix=h5f; fi
data_dir=${data_dir}.${data_suffix}

# Set up command line
if $wass; then wass_cmd='--wasserstein'; else wass_cmd=''; fi
if $use_noise; then use_noise_cmd='--use_noise'; else use_noise_cmd=''; fi
if $retrain; then retrain_cmd='--retrain'; else retrain_cmd=''; fi

if $marcc; then
  cmd_prefix="$HOME/costar_plan/costar_models/scripts/"
else
  cmd_prefix='rosrun costar_models '
fi

## Pretrain_image_encoder ---------------------------

if ! $skip_encoder; then
  status_file="$MODELDIR"/status_pretrain.txt
  # Check if we should load the model
  load_cmd=''
  $load_model || ($resume && [[ -f $status_file ]]) && load_cmd='--load_model'

  # Calculate epochs left
  epochs_done=0
  if $resume && [[ -f $status_file ]]; then
    contents=($(cat $status_file))
    epochs_done=${contents[0]}
  fi
  # Check for resume after finish
  if $resume && (($epochs_done >= $epochs1)); then
    echo Skipping pretrain_image due to resume
    echo $epochs_done/$epochs1 epochs already done
  elif $gan_encoder; then
    echo "Training gan encoder. $epochs_done/$epochs1 epochs done"
    ${cmd_prefix}ctp_model_tool \
      --features $features \
      -e $epochs1 \
      --initial_epoch $epochs_done \
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
      --unique_id _pretrain \
      $wass_cmd \
      $load_cmd
  else
    echo "Training non-gan image encoder. $epochs_done/$epochs1 epochs done"
    ${cmd_prefix}ctp_model_tool \
      --features $features \
      -e $epochs1 \
      --initial_epoch $epochs_done \
      --model pretrain_image_encoder \
      --data_file $data_dir \
      --lr $lr \
      --dropout_rate $dropout \
      --model_directory $MODELDIR/ \
      --optimizer $optimizer \
      --steps_per_epoch 300 \
      --noise_dim $noise_dim \
      --loss $loss \
      --batch_size 64 \
      --no_disc \
      --unique_id _pretrain \
      $load_cmd
  fi
fi

## Conditional gan ---------------------------------

if ! $skip_cond; then

  status_file="$MODELDIR"/status_cond.txt
  # Check if we should load the model
  load_cmd=''
  $load_model || ($resume && [[ -f $status_file ]]) && load_cmd='--load_model'

  req_dir_cmd=''
  [[ $enc_dir ]] && req_dir_cmd="--req_directory $enc_dir"

  # Calculate epochs left
  epochs_done=0
  if $resume && [[ -f $status_file ]]; then
    contents=($(cat $status_file))
    epochs_done=${contents[0]}
  fi
  # Check for resume after finish
  if $resume && (($epochs_done >= $epochs2)); then
    echo Skipping conditional gan due to resume!
  else
    echo "Training conditional gan. $epochs_done/$epochs2 epochs done."
      ${cmd_prefix}ctp_model_tool \
      --features $features \
      -e $epochs2 \
      --initial_epoch $epochs_done \
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
      --unique_id _cond \
      $wass_cmd \
      $use_noise_cmd \
      $load_cmd \
      $req_dir_cmd
  fi
fi
