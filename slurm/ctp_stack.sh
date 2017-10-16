#!/bin/bash -l
#SBATCH --job-name=stack1
#SBATCH --time=0-48:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu

set -e
set -x
set -u

echo "Running $@ on $SLURMD_NODENAME ..."

module load tensorflow/cuda-8.0/r1.3 

# models_stack: no dropouts, 2 tforms
# models_stack2: all dropouts, including on tforms
# models_stack3: no dropouts on tforms?
# models_stack4: smaller hidden arm+gripper layer and smaller hidden layers
# models_stack5: (8,8,64) and ph output
# models_stack6: (8,8,64) and noise
# sequence A:(8,8,64), options, 1 transform layer
# sequence B:(4,4,64), options, 2 transform layers
# sequence C:(4,4,64), options, 3 transform layers
# sequence D:(8,8,64), options, 3 transform layers, dense
# sequence F: dense, 128
# sequence G: dense, 32
# sequence H: dense, 32, 1 tform
# sequence I: dense, bigger
# sequence J: dense, 256, multiple tforms
$HOME/costar_plan/costar_models/scripts/ctp_model_tool \
	--features multi \
	-e 100 \
	--model predictor \
	--data_file $HOME/work/ctp_rpy.npz \
	--lr 0.001 \
	--model_directory $HOME/.costar/models_stackO/ \
	--optimizer adam \
  --upsampling conv_transpose \
  --hypothesis_dropout false \
  --dropout_rate 0.5 \
  --use_noise true \
  --steps_per_epoch 300 \
  --noise_dim 8 \
	--batch_size 64

