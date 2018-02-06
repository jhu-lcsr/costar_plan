#!/bin/bash -l

set -e
set -x
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

OPTS=$(getopt --long retrain,load_model -n start_ctp_gans -- "$@")

retrain=false
load_model=false
lr=0.0001
dr=0.1
opt=adam
noise_dim=4
wass=wass
loss=mae

while true; do
  case "$1" in
    --retrain) retrain=true; shift ;;
    --) shift; break ;;
    *) echo "Internal error!" ; exit 1 ;;
  esac
done

retrain_cmd=''
$retrain && retrain_cmd='--retrain'

for wass_cmd in --wass ''; do

  [[ $wass ]] && opt=rmsprop

  for noise_cmd in --noise ''; do
    sbatch "$SCRIPT_DIR"/ctp_gan.sh ctp_dec        multi   --lr $lr --dr $dr --opt $opt $wass_cmd $noise_cmd $retrain_cmd
    sbatch "$SCRIPT_DIR"/ctp_gan.sh husky_data     husky   --lr $lr --dr $dr --opt $opt $wass_cmd $noise_cmd $retrain_cmd
    sbatch "$SCRIPT_DIR"/ctp_gan.sh suturing_data2 jigsaws --lr $lr --dr $dr --opt $opt $wass_cmd $noise_cmd $retrain_cmd
  done
done
