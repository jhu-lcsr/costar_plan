#!/bin/bash -l
#SBATCH --job-name=ctpY
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu

set -e
set -x
set -u

echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

export DATASET="ctp_dec"
export MODELDIR="$HOME/.costar/models_stack_Y$1$3$2$4$5$6"

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model predictor \
	--data_file $HOME/work/$DATASET.h5f \
	--lr $1 \
	--dropout_rate $2 \
	--decoder_dropout_rate $2 \
  --model_directory $MODELDIR/ \
	--optimizer $3 \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $5 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --skip_connections $6 \
  --batch_size 32
  #--success_only \

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model pretrain_image_encoder \
	--data_file $HOME/work/$DATASET.npz \
	--lr $1 \
	--dropout_rate $2 \
	--decoder_dropout_rate $2 \
  --model_directory $MODELDIR/ \
	--optimizer $3 \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $5 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --skip_connections $6 \
  --batch_size 64

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model pretrain_state_encoder \
	--data_file $HOME/work/$DATASET.npz \
	--lr $1 \
	--dropout_rate $2 \
	--decoder_dropout_rate $2 \
  --model_directory $MODELDIR/ \
	--optimizer $3 \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $5 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --skip_connections $6 \
  --batch_size 128
 
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model predictor2 \
	--data_file $HOME/work/$DATASET.npz \
	--lr $1 \
	--dropout_rate $2 \
	--decoder_dropout_rate $2 \
  --model_directory $MODELDIR/ \
	--optimizer $3 \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $5 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --skip_connections $6 \
  --batch_size 32
  #--success_only \

