
export DATASET="ctp_dec"
export MODELDIR="$HOME/.costar/models_stack_Y$1$3$2$4$5$6"

$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model pretrain_sampler \
	--data_file $HOME/work/$DATASET.h5f \
	--lr 0.001 \
	--dropout_rate 0.2 \
	--decoder_dropout_rate 0.2 \
  --model_directory $MODELDIR/ \
	--optimizer adam \
  --use_noise true \
  --steps_per_epoch 500 \
  --noise_dim 0 \
  --hypothesis_dropout 1 \
  --upsampling conv_transpose \
  --skip_connections 0 \
  --batch_size 64
 
