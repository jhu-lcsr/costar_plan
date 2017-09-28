#!/bin/bash -l
#SBATCH --job-name=ctp_stack
#SBATCH --time=0-48:0:0
#SBATCH --nodes=1
#SBATCH -p unlimited
#SBATCH -g 2
#SBATCH --cpus-per-task=6
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
	-e 100 \
	--model predictor \
	--data_file $HOME/work/ctp_test2.npz \
	--lr 0.001 \
	 --model_directory $HOME/.costar/models_stack/ \
	--optimizer adam \
	--batch_size 64

