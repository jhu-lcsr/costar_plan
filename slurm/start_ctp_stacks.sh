#!/bin/bash -l

set -e
set -x
set -u

module load tensorflow/cuda-8.0/r1.3 

for $i in {1..10}
do
	# start the script
	$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
		--features multi \
		-e 100 \
		--model predictor \
		--data_file $HOME/work/ctp_test2.npz \
		--lr 0.001 \
		 --model_directory $HOME/.costar/models_stack/ \
		--optimizer adam \
		--batch_size 64
done
