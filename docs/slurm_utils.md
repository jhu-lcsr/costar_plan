
# Using SLURM with CTP

## Setup

Source slurm utilities script:
```
# Change this to whatever seems appropriate
export COSTAR_PLAN_DIR=$HOME/costar_plan

source $COSTAR_PLAN_DIR/slurm/slurm_utils.sh
```

## Running Jobs

```
running_jobs
```

Example output:
```
24331573 2:01:22 ctp_dec_0.0001_rmsprop_0.1_1_mae_wass_nonoise_noganenc_noretrain
24331574 2:01:22 husky_data_0.0001_rmsprop_0.1_1_mae_wass_nonoise_noganenc_noretrain
24331575 2:01:22 suturing_data2_0.0001_rmsprop_0.1_1_mae_wass_nonoise_noganenc_noretrain
24331576 2:01:22 ctp_dec_0.0001_adam_0.1_1_mae_nowass_noise_ganenc_noretrain
24331577 2:01:22 husky_data_0.0001_adam_0.1_1_mae_nowass_noise_ganenc_noretrain
24331578 2:01:22 suturing_data2_0.0001_adam_0.1_1_mae_nowass_noise_ganenc_noretrain
24331579 2:01:22 ctp_dec_0.0001_adam_0.1_1_mae_nowass_noise_noganenc_noretrain
24331580 2:01:22 husky_data_0.0001_adam_0.1_1_mae_nowass_noise_noganenc_noretrain
77 running jobs
```

This lists the time and the current number of the job as well as the output directory.

## Contact

Yotam Barnoy
