#!/usr/bin/env bash
#rosrun costar_models ctp_model_tool \
#	--features multi -i 50000 -e 100 \
#	--model predictor --data_file ~/datasets/ctp_test.npz \
#	--si 0 --lr 0.001 --optimizer nadam \
#        --batch_size 64 

export MODELDIR=$HOME/.costar/models
export LR=0.001
export DR=0.2
export OPT=adam
export ND=0
export HDR=0.2


if [ 0 -gt 1 ]
then
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_image_encoder \
    --data_file $DATASET\
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

if [ 2 -gt 1 ]
then
  rosrun costar_models ctp_model_tool \
    --features multi \
    -e 100 \
    --model pretrain_sampler \
    --data_file $HOME/work/$DATASET.h5f \
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


rosrun costar_models ctp_model_tool \
  --features multi \
  -e 100 \
  --model conditional_sampler2 \
  --data_file $HOME/work/$DATASET.h5f \
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


rosrun costar_models ctp_model_tool \
  --features multi \
  -e 100 \
  --model predictor2 \
  --data_file $HOME/work/$DATASET.h5f \
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

