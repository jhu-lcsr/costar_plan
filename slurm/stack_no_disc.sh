#!/bin/bash -l
#SBATCH --job-name=stack_no_disc
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu


retrain=false
use_disc=false
lr=0.001
dr=0.1
opt=adam
noise_dim=4
wass=wass
loss=mae
./ctp.sh $lr $dr $opt $noise_dim $loss $retrain $use_disc
