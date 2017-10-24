#!/bin/bash -l
#SBATCH --job-name=ctpS
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
	-e 100 \
	--model predictor \
	--data_file $HOME/work/ctp_rpy.npz \
	--lr $1 \
	--dropout_rate 0.125 \
	--decoder_dropout_rate $2 \
  --model_directory $HOME/.costar/models_stack_S2_$1$3$2$4$5/ \
	--optimizer $3 \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $5 \
  --hypothesis_dropout $4 \
  --upsampling conv_transpose \
  --batch_size 32

#--success_only \

