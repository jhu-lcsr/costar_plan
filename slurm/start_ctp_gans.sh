#!/bin/bash -l

set -e
set -x
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

OPTS=$(getopt -o '' --long retrain,load_model,gan_encoder,skip_encoder,suffix:,no_resume,epochs1:,epochs2: -n start_ctp_gans -- "$@")

[[ $? != 0 ]] && echo "Failed parsing options." && exit 1

echo "$OPTS"
eval set -- "$OPTS"

retrain=false
load_model=false
lr=0.0001
dr=0.1
opt=adam
noise_dim=4
wass=wass
loss=mae
gan_encoder=false
skip_encoder=false
suffix=''
resume=true # resume a job
epochs1=200
epochs2=200

while true; do
  case "$1" in
    --retrain) retrain=true; shift ;;
    --encoder) gan_encoder=false; shift ;;
    --gan_encoder) gan_encoder=true; shift ;;
    --skip_encoder) skip_encoder=true; shift ;;
    --load_model) load_model=true; shift ;;
    --suffix) suffix="$2"; shift 2 ;;
    --no_resume) resume=false; shift ;;
    --epochs1) epochs1="$2"; shift 2 ;;
    --epochs2) epochs2="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Internal error!" ; exit 1 ;;
  esac
done

#if $retrain; then retrains=--retrain; else retrains=--retrain ''; fi
#if $gan_encoder; then gan_cmd='--gan_encoder'; else gan_cmd=''; fi
if $skip_encoder; then skip_cmd='--skip_encoder'; else skip_cmd=''; fi
if $load_model; then load_cmd='--load_model'; else load_cmd=''; fi
if [[ $suffix != '' ]]; then suffix_cmd="--suffix $suffix"; else suffix_cmd=''; fi
if $resume; then resume_cmd=''; else resume_cmd='--no_resume'; fi

for wass_cmd in --wass ''; do
  if [[ $wass_cmd == '--wass' ]]; then opt=rmsprop; else opt=adam; fi
  for noise_cmd in --noise ''; do
    for gan_cmd in --gan_encoder ''; do
      # double the epochs for gan encoder
      [[ $gan_cmd == '--gan_encoder' ]] && epochs1=$(($epochs1 * 2))

      for retrain_cmd in --retrain ''; do
        function call_ctp() {
          sbatch "$SCRIPT_DIR"/ctp_gan.sh "$1" "$2" --lr $lr --dr $dr \
            --opt $opt --epochs1 $epochs1 --epochs2 $epochs2 \
            $wass_cmd $noise_cmd $retrain_cmd $gan_cmd \
            $load_cmd $skip_cmd $suffix_cmd $resume_cmd
        }
        call_ctp ctp_dec multi
        call_ctp husky_data husky
        call_ctp suturing_data2 jigsaws
      done

      [[ $gan_cmd == '--gan_encoder' ]] && epochs1=$(($epochs1 / 2))
    done
  done
done
