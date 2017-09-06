#!/bin/bash -l
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

set -e
set -x
set -u

#sbatch -n 6 -p unlimited -g 1 --time=24:0:0 $HOME/costar_plan/slurm/$1
sbatch $HOME/costar_plan/slurm/$1
