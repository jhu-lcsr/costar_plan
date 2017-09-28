#!/bin/bash -l
#SBATCH --job-name=predictor250blocksmove
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
	-i 1000 -e 100 \
	--model predictor \
	 --data_file $HOME/work/ctp_blocks_500b.npz \
	 --lr 0.001 \
	 --model_directory $HOME/.costar/models2/ \
	--optimizer adam \
	--batch_size 64 
