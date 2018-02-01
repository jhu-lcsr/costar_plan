#!/bin/bash -l

set -e
set -x
set -u

lr=0.001
dr=0.1
opt=adam
noise_dim=4
wass=wass
loss=mae

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

# Start training things
retrain=false
use_disc=true
sbatch ctp_suturing.sh $lr $dr $opt $noise_dim $loss $retrain $use_disc

retrain=false
use_disc=false
sbatch ctp_suturing.sh $lr $dr $opt $noise_dim $loss $retrain $use_disc

retrain=true
use_disc=true
sbatch ctp_suturing.sh $lr $dr $opt $noise_dim $loss $retrain $use_disc

lr=0.0001
dr=0.1
opt=adam
for w in wass nowass; do
  for t in true false; do
    sbatch "$SCRIPT_DIR"/ctp_gan.sh suturing_data2 jigsaws $lr $dr $opt $noise_dim $loss $w $t
  done
done
