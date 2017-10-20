#!/bin/bash -l
#SBATCH --job-name=stack2
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
	--lr 0.001 \
  --model_directory $HOME/.costar/models_stackR_no_sample/ \
	--optimizer adam \
  --dropout_rate 0.5 \
  --decoder_dropout_rate 0.125 \
  --upsampling conv_transpose \
  --hypothesis_dropout false \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim 8 \
  --success_only \
  --batch_size 32

