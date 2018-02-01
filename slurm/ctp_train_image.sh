#!/bin/bash -l
#SBATCH --job-name=Nimageonly
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
	--data_file $HOME/work/ctp_rpy.npz \
  --model predictor \
  -e 100 \
  --features multi \
  --optimizer adam \
  --lr 0.001 \
  --model_directory $HOME/.costar/models_N_image/ \
  --upsampling conv_transpose \
  --use_noise \
  --noise_dim 8 \
  --steps_per_epoch 300  \
  --dropout_rate 0.5 \
	--batch_size 64

