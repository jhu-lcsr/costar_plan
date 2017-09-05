#!/bin/bash -l
set -e
set -x
set -u

#SBATCH --job-name=b500
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

$HOME/costar_plan/costar_models/scripts/ctp_model_tool --features multi -i 1000 -e 1000 --model predictor --data_file $HOME/work/ctp_blocks_500.npz --lr 0.001  --model_directory $HOME/.costar/models3/ --optimizer nadam
