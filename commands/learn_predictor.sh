#!/usr/bin/env bash
#rosrun costar_models ctp_model_tool \
#	--features multi -i 50000 -e 100 \
#	--model predictor --data_file ~/datasets/ctp_test.npz \
#	--si 0 --lr 0.001 --optimizer nadam \
#        --batch_size 64 

export MODELDIR="$HOME/.costar/models"
export LR=0.001
export DR=0.2
export OPT=adam
export ND=0
export HDR=0.2
export DATASET=test2

if true; then
  echo "---- Image Representation ----"
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file ${DATASET}.h5f\
    --lr $LR \
    --dropout_rate $DR \
    --decoder_dropout_rate $DR \
    --model_directory $MODELDIR/ \
    --optimizer $OPT \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $ND \
    --hypothesis_dropout $HDR \
    --upsampling conv_transpose \
    --batch_size 64
fi

if false; then
  echo "---- Image + Extra data representation ----"
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_sampler \
    --data_file ${DATASET}.h5f \
    --lr $LR \
    --dropout_rate $DR \
    --decoder_dropout_rate $DR \
    --model_directory $MODELDIR/ \
    --optimizer $OPT \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $ND \
    --hypothesis_dropout $HDR \
    --upsampling conv_transpose \
    --batch_size 64
fi


if false; then
  echo "---- Transitions + extra data ----"
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model conditional_sampler2 \
    --data_file ${DATASET}.h5f \
    --lr $LR \
    --dropout_rate $DR \
    --decoder_dropout_rate $DR \
    --model_directory $MODELDIR/ \
    --optimizer $OPT \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $ND \
    --hypothesis_dropout $HDR \
    --upsampling conv_transpose \
    --batch_size 64
fi

# transitions, only images
echo "---- Transitions with only images ----"
rosrun costar_models ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_image \
  --data_file ${DATASET}.h5f \
  --lr $LR \
  --dropout_rate $DR \
  --decoder_dropout_rate $DR \
  --model_directory $MODELDIR/ \
  --optimizer $OPT \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim $ND \
  --hypothesis_dropout $HDR \
  --upsampling conv_transpose \
  --batch_size 64


# multiple parallel predictions
if false; then
  echo "---- Transitions with images + data ----"
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model predictor2 \
    --data_file ${DATASET}.h5f \
    --lr $LR \
    --dropout_rate $DR \
    --decoder_dropout_rate $DR \
    --model_directory $MODELDIR/ \
    --optimizer $OPT \
    --use_noise true \
    --steps_per_epoch 500 \
    --noise_dim $ND \
    --hypothesis_dropout $HDR \
    --upsampling conv_transpose \
    --batch_size 32 # --retrain 
    #--success_only \
fi

