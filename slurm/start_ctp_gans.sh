#!/bin/bash -l

set -e
set -x
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

lr=0.0001
dr=0.1
opt=rmsprop
noise_dim=4
wass=wass
loss=mae
lr=0.0001

for w in wass nowass; do
  for t in true false; do
    sbatch "$SCRIPT_DIR"/ctp_gan.sh ctp_dec multi $lr $dr $opt $noise_dim $loss $w $t
    sbatch "$SCRIPT_DIR"/ctp_gan.sh husky_data husky $lr $dr $opt $noise_dim $loss $w $t
    sbatch "$SCRIPT_DIR"/ctp_gan.sh suturing_data2 jigsaws $lr $dr $opt $noise_dim $loss $w $t
  done
done
