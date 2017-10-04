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

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

module load tensorflow/cuda-8.0/r1.3 

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 250 \
	--model predictor \
	--data_file $HOME/work/ctp_value.npz \
	--lr 0.001 \
	--model_directory $HOME/.costar/models_stackA6_hdtrue/ \
	--optimizer adam \
  --dropout_rate 0.5 \
  --decoder_dropout_rate 0.125 \
  --upsampling conv_transpose \
  --hypothesis_dropout false \
	--dropout_rate 0.2 \
  --use_noise true \
  --noise_dim 32 \
	--batch_size 64

