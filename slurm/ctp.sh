#!/bin/bash -l
#SBATCH --job-name=ctpF
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

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 250 \
	--model predictor \
	--data_file $HOME/work/ctp_value2.npz \
	--lr $1 \
	--dropout_rate 0.5 \
	--decoder_dropout_rate $2 \
	--model_directory $HOME/.costar/models_stack_G$1$3$2$4/ \
	--optimizer $3 \
  --use_noise true \
  --noise_dim 32 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --batch_size 64

