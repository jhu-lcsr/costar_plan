#!/usr/bin/env bash
echo "------------------------------------"
echo "Running $0"
echo "This will train the following:"
echo " - V(x)"
echo " - p(action | x)"
echo " - pi(x, action) ~ goal_pose"
echo " - pi(x, action) ~ u"
echo "------------------------------------"
rosrun costar_models ctp_model_tool \
  --model secondary --data_file data.h5f \
  --epochs 50 \
  --lr 0.001  --dropout_rate 0.1 \
  --features multi --batch_size 64 \
  --steps_per_epoch 500 --submodel value $1

rosrun costar_models ctp_model_tool \
  --model secondary --data_file data.h5f \
  --epochs 50 \
  --lr 0.001  --dropout_rate 0.1 --features multi \
  --batch_size 64 --steps_per_epoch 500 --submodel next $1

rosrun costar_models ctp_model_tool \
  --model secondary --data_file data.h5f \
  --epochs 100 \
  --lr 0.001  --dropout_rate 0.1 \
  --features multi --batch_size 64 \
  --steps_per_epoch 500 --submodel actor $1

 rosrun costar_models ctp_model_tool \
   --model secondary --data_file data.h5f \
  --epochs 200 \
   --lr 0.001  --dropout_rate 0.1 \
   --features multi --batch_size 64 \
   --steps_per_epoch 500 --submodel pose $1
