#!/bin/bash -l

set -e
set -x
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"/../costar_models/python
python setup.py install --user
cd -

OPTS=$(getopt -o '' --long retrain,load_model,gan_encoder,no_gan_encoder,skip_encoder,suffix:,no_resume,epochs1:,epochs2:,skip_cond,no_husky,no_jigsaws,no_ctp,no_wass,common_encoder,dense_transform -n start_ctp_gans -- "$@")

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
epochs1=100
epochs2=100
skip_cond_cmd=''
run_ctp=true
run_husky=true
run_jigsaws=true
run_wass=true
common_encoder=false # Use an encoder in a separate location
dense_transform=false

while true; do
  case "$1" in
    --retrain) retrain=true; shift ;;
    --no_gan_encoder) gan_encoder=false; shift ;;
    --gan_encoder) gan_encoder=true; shift ;;
    --skip_encoder) skip_encoder=true; shift ;;
    --common_encoder) common_encoder=true; skip_encoder=true; shift ;;
    --skip_cond) skip_cond_cmd='--skip_cond'; shift ;;
    --load_model) load_model=true; shift ;;
    --suffix) suffix="$2"; shift 2 ;;
    --no_resume) resume=false; shift ;;
    --epochs1) epochs1="$2"; shift 2 ;;
    --epochs2) epochs2="$2"; shift 2 ;;
    --no_ctp) run_ctp=false; shift ;;
    --no_husky) run_husky=false; shift ;;
    --no_jigsaws) run_jigsaws=false; shift ;;
    --no_wass) run_wass=false; shift ;;
    --dense_transform) dense_transform=true ;;
    --) shift; break ;;
    *) echo "Internal error!" ; exit 1 ;;
  esac
done

if $skip_encoder; then skip_cmd='--skip_encoder'; else skip_cmd=''; fi
if $load_model; then load_cmd='--load_model'; else load_cmd=''; fi
if [[ $suffix != '' ]]; then suffix_cmd="--suffix $suffix"; else suffix_cmd=''; fi
if $resume; then resume_cmd=''; else resume_cmd='--no_resume'; fi
# Check for the common encoder location
if $common_encoder; then  enc_dir_cmd='--enc_dir ~/.costar/pretrain_codec'; else enc_dir_cmd=''; fi
if $dense_transform; then dense_transform_cmd='--dense_transform'; else dense_transform_cmd=''; fi

if $run_wass; then wass_list='true false'; else wass_list=false; fi
for do_wass in $wass_list; do
  if $do_wass; then wass_cmd='--wass'; else wass_cmd=''; fi
  if $do_wass; then opt=rmsprop; else opt=adam; fi
  for noise_cmd in --noise ''; do
    if $gan_encoder; then enc_list='true false'; else enc_list='false'; fi
    for do_gan_enc in $enc_list; do
      if $do_gan_enc; then enc_cmd='--gan_encoder'; else enc_cmd=''; fi
      # double the epochs for gan encoder
      $do_gan_enc && epochs1=$(($epochs1 * 2))

      for retrain_cmd in ''; do # --retrain
        function call() {
          sbatch "$SCRIPT_DIR"/ctp_gan.sh "$1" "$2" --lr $lr --dr $dr \
            --opt $opt --epochs1 $epochs1 --epochs2 $epochs2 \
            $wass_cmd $noise_cmd $retrain_cmd $enc_cmd \
            $load_cmd $skip_cmd $suffix_cmd $resume_cmd $skip_cond_cmd \
            $enc_dir_cmd $dense_transform_cmd
        }
        $run_ctp && call ctp_dec multi
        $run_husky && call husky_data husky
        $run_jigsaws && call suturing_data2 jigsaws
      done

      $do_gan_enc && epochs1=$(($epochs1 / 2))
    done
  done
done
