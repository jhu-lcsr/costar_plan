#!/bin/bash -l
#SBATCH --job-name=gposeS
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

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

module load tensorflow/cuda-8.0/r1.3 


$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
  --features multi \
	-e 500 \
	--model goal_sampler \
	--data_file $HOME/work/ctp_rpy.npz \
	--lr 0.001 \
	--model_directory $HOME/.costar/models_stackS_goal_pose_only/ \
	--optimizer adam \
  --upsampling conv_transpose \
  --hypothesis_dropout false \
  --dropout_rate 0.001 \
  --use_noise true \
  --noise_dim 32 \
  #--success_only \
	--batch_size 64
