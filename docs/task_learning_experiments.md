
# Experiments and Training Notes for Task Learning

This document is for experiments from 2018-01-05. 

## Learning

### Example Training Command

### Training On MARCC

MARCC is our cluster for machine learning, equipped with a large set of Tesla K80 GPUs. We assume that when training on a cluster like MARCC, you will not want a full ROS workspace, so instead we assume you will install to some path $COSTAR_PLAN and just run scripts.

To run on MARCC, the easiest set up is always:
```
export COSTAR_PLAN=$HOME/costar_plan
$COSTAR_PLAN/slurm/start_ctp_stacks.sh
```

This will run the `$COSTAR_PLAN/slurm/ctp.sh$ script with a few different arguments to start SLURM jobs.


## Validation

### Hidden State

```
rosrun costar_models ctp_hidden.py --cpu --model conditional_image --data_file test2.h5f
```

### Transformation

```
rosrun costar_models ctp_transform.py --cpu --model conditional_image --data_file test2.h5f
```

