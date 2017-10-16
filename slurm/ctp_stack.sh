#!/bin/bash -l
#SBATCH --job-name=stack1
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
	--model_directory $HOME/.costar/models_stackM/ \
	--optimizer adam \
  --upsampling conv_transpose \
  --hypothesis_dropout false \
  --dropout_rate 0.5 \
  --use_noise false \
  --steps_per_epoch 300 \
  --noise_dim 8 \
  --sampling \
	--batch_size 64

