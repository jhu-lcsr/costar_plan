
#!/usr/bin/env bash

rosrun costar_bullet gui_start \
  --robot ur5 --task stack1 --agent task \
  -i 5000 --features multi  --verbose \
  --seed 0 \
  --cpu \
  --save --data_file test.h5f \
  --collection_mode goal

# NOTE: removing this flag now that we are predicting both successful and
# unsuccessful futures from any given state.
#--success_only \

