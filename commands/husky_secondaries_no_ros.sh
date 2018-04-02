#!/usr/bin/env bash
echo "------------------------------------"
echo "Running $0"
echo "This will train the following:"
echo " - V(x)"
echo " - p(action | x)"
echo " - pi(x, action) ~ goal_pose"
echo " - pi(x, action) ~ u"
echo "------------------------------------"
./costar_models/scripts/ctp_model_tool \
  --model secondary --data_file husky_data.npz \
  --epochs 50 \
  --lr 0.001  --dropout_rate 0.1 \
  --features husky --batch_size 64 \
  --steps_per_epoch 500 --submodel value $1

./costar_models/scripts/ctp_model_tool \
  --model secondary --data_file husky_data.npz \
  --epochs 50 \
  --lr 0.001  --dropout_rate 0.1 --features husky \
  --batch_size 64 --steps_per_epoch 500 --submodel next $1

./costar_models/scripts/ctp_model_tool \
  --model secondary --data_file husky_data.npz \
  --epochs 50 \
  --lr 0.001  --dropout_rate 0.1 --features husky \
  --batch_size 64 --steps_per_epoch 500 --submodel q $1

./costar_models/scripts/ctp_model_tool \
  --model secondary --data_file husky_data.npz \
  --epochs 100 \
  --lr 0.001  --dropout_rate 0.1 \
  --success_only \
  --features husky --batch_size 64 \
  --steps_per_epoch 500 --submodel actor $1

 ./costar_models/scripts/ctp_model_tool \
   --model secondary --data_file husky_data.npz \
   --epochs 200 \
   --lr 0.001  --dropout_rate 0.1 \
   --features husky --batch_size 64 \
   --success_only \
   --steps_per_epoch 500 --submodel pose $1

