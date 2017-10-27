import sys
if len(sys.argv) > 1:
    matchstr = sys.argv[1]

listvar = $(ls).split('\n')
for weightsfile in listvar:
    # print(weightsfile)
    if '.h5' in weightsfile and matchstr in weightsfile and 'evaluation_dataset' not in weightsfile:
        print(weightsfile)
        # echo @(weightsfile)
        CUDA_VISIBLE_DEVICES="1" && python grasp_train.py --random_crop=1 --batch_size=20 --epochs 100 --load_weights @(weightsfile) --pipeline_stage eval --grasp_model grasp_model_resnet --data_dir=~/datasets/grasping.ssd/