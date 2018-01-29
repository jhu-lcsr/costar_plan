#!/bin/bash -l

set -e
set -x
set -u

if [[ $# < 1 ]]; then
  echo Usage: $0 wass/nowass
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

wass=$1

lrs='0.001 0.0002 0.0001'

if [[ $wass == wass* ]]; then 
  lrs='0.00005 0.0001 0.00002'
fi

for lr in $lrs; do
  # just use the adam optimizer
  for opt in adam; do
    for loss in mae; do #logcosh
    # what do we do about skip connections?
    for skip in 0; do # 1
      # Noise: add extra ones with no noise at all
      for noise_dim in 0; do # 1 8 32
        for dr in 0. 0.1 0.2; do # 0.3 0.4 0.5; do
          echo "starting cpt_dec multi $wass LR=$lr, dr=$dr, opt=$opt, noise=$noise_dim"
          sbatch "$SCRIPT_DIR"/ctp_gan.sh ctp_dec multi $lr $dr $opt $noise_dim $loss $wass
        done
      done
    done
    done
  done
done

