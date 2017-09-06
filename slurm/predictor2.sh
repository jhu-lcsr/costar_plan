#!/bin/bash -l
#SBATCH --job-name=predictor250blocksmove
#SBATCH --time=0-24:0:0
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu
#SBATCH -p unlimited
#SBATCH -g 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G


set -e
set -x
set -u

echo
echo "Running $@ on $SLURMD_NODENAME ..."
echo

$HOME/costar_plan/costar_models/scripts/ctp_model_tool --features multi -i 1000 -e 100 --model predictor --data_file $HOME/work/ctp_blocks_500b.npz --lr 0.001  --model_directory $HOME/.costar/models2/ --optimizer nadam --load_model
