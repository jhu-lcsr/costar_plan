#!/usr/bin/env bash
cp $HOME/.costar/models/goal_sampler_model_train_predictor.h5f \
  $HOME/.costar/models/goal_sampler_model_predictor_weights_bkup.h5f
cp $HOME/.costar/models/goal_sampler_model_predictor_weights.h5f \
  $HOME/.costar/models/goal_sampler_model_train_predictor.h5f 
