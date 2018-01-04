#!/bin/bash -l
#SBATCH --job-name=ctpZ
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

export NUM_ARGS=3

if [[ "$#" < $NUM_ARGS ]]; then
	echo "Usage: $0 lr gan_method epochs"
	exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export DATASET="dev_yb/test2"
export LR=$1
export GAN_METHOD=$2
export EPOCHS=$3
export DR=0.1
export HD=0.1
export NOISE_DIM=0
export SKIP=0
export LOSS=logcosh
export OPTIM=adam

export MODELDIR="$HOME/.costar/models_gan_$1$2_${SLURM_JOB_ID}"

if true
then
  $SCRIPT_DIR/../costar_models/scripts/ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_image_gan \
    --data_file $HOME/work/$DATASET.h5f \
    --lr $LR \
    --dropout_rate $DR \
    --decoder_dropout_rate $DR \
    --model_directory $MODELDIR/ \
    --optimizer $OPTIM \
    --use_noise false \
    --steps_per_epoch 500 \
    --noise_dim 0 \
    --hypothesis_dropout 1 \
    --upsampling conv_transpose \
    --skip_connections 0 \
    --loss $LOSS \
    --batch_size 64 \
		--clipnorm 10
fi

