
# Google Brain Grasp Dataset APIs

Author and maintainer: `Andrew Hundt <ATHundt@gmail.com>`

<img width="1511" alt="2017-12-16 surface relative transforms correct" src="https://user-images.githubusercontent.com/55744/34134058-5846b59e-e426-11e7-92d6-699883199255.png">
This version should be ready to use when generating data real training.

Plus now there is a flag to draw a circle at the location of the gripper as stored in the dataset:
![102_grasp_0_rgb_success_1](https://user-images.githubusercontent.com/55744/34133964-ccf57caa-e425-11e7-8ab1-6bba459a5408.gif)

A new feature is writing out depth image gifs:
![102_grasp_0_depth_success_1](https://user-images.githubusercontent.com/55744/34133966-d0951f28-e425-11e7-85d1-aa2706a4ba05.gif)

Image data can be resized:

![102_grasp_1_rgb_success_1](https://user-images.githubusercontent.com/55744/34430739-3adbd65c-ec36-11e7-84b5-3c3712949914.gif)

The blue circle is a visualization, not actual input, which marks the gripper stored in the dataset pose information.

Color augmentation is also available:

![102_grasp_2_rgb_success_1](https://user-images.githubusercontent.com/55744/34698561-ba2bd61e-f4a6-11e7-88d9-5091aed500fe.gif)
![102_grasp_3_rgb_success_1](https://user-images.githubusercontent.com/55744/34698564-bef2fba0-f4a6-11e7-9547-06b4410d86aa.gif)

### How to view the vrep dataset visualization

1. copy the .ttt file and the .so file (.dylib on mac) into the `costar_google_brainrobotdata/vrep` folder.
2. Run vrep with -s file pointing to the example:
```
./vrep.sh -s ~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/vrep/kukaRemoteApiCommandServerExample.ttt
```
4. vrep should load and start the simulation
5. make sure the folder holding `vrep_grasp.py` is on your PYTHONPATH
6. cd to `~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/`, or wherever you put the repository
7. run `export CUDA_VISIBLE_DEVICES="" && python2 vrep_grasp.py`

## Hyperparameter search


### Google Brain Grasping Dataset

To run the search execute the following command

```
export CUDA_VISIBLE_DEVICES="0" && python2 google_grasp_hyperopt.py --run_name single_prediction_all_transforms
```

Generating a hyperparameter search results summary for google brain grasping dataset classification:

```
python hyperopt_rank.py --log_dir hyperopt_logs_google_brain_classification --sort_by val_acc
```

### Cornell Dataset

These are instructions for training on the [cornell grasping dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php).


#### Downloading the dataset

To download the dataset and generate the tensorflow tfrecord dataset files simply run the following command:

```
python cornell_grasp_dataset_writer.py
```

This should take about 5-6 GB of space, and by default the dataset files will go in:

```

flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
```

#### Where results can be found

Check the value of the following parameter with each command:

```--log_dir```

Files will be created containing training results in that file.

By default it is `./logs_cornell/`:

#### Regression

Regression tries to predict a grasp position and orientation based on an image alone.

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_grasp_train_regression.py --pipeline_stage k_fold --run_name 60_epoch_real_run
```

The following commands will generate a hyperopt summary.

Generating a hyperparameter search results summary for cornell regression

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_regression --sort_by val_grasp_jaccard
```

Please note that there are currently bugs in the loss with the regression problem. Additonally,
a custom command must be run to find the correct intersection over union score.

#### Classification

Takes a grasp position, orientation, gripper width, etc as input, then returns a score indicating if it will be a successful grasp of an object or a failed grasp.
Given the grasp command input, the proposed grasp image
is rotated and centered on the grasp, so that all grasps have the gripper
in a left-right orientation to the image and the gripper is at the center.

 - `cornell_hyperopt.py` is basically a configuration file used to configure hyperparameter optimization for the cornell grasping dataset.
 - `hyperopt.py` is another configuration file that sets up the range of models you should search during hyperopt.
     - shared by google and cornell for both classification and regression training
     - you may also want to modify this to change what kind of models are being searched.

In `cornell_hyperopt.py` make sure the following variable is set correctly:

```
    FLAGS.problem_type = 'classification'
```

This will run hyperopt search for a couple of days on a GTX 1080 or Titan X to find the best model.

Hyperparameter search to find the best model:

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_hyperopt.py --log_dir hyperopt_logs_cornell_classification
```

Result will be in files with names like these:

```

2018-05-08-17-55-00__bayesian_optimization_convergence_plot.png
2018-05-08-17-55-00__hyperoptions.json
2018-05-08-17-55-00__optimized_hyperparams.json
2018-05-23-19-08-31__hyperoptions.json
```


Generating a hyperparameter search results summary for cornell classification:

```
python hyperopt_rank.py --log_dir hyperopt_logs_cornell_classification --sort_by val_binary_accuracy
```

This will generate a hyperopt ranking file:

```
hyperopt_rank.csv
```

Look at the CSV file above, and it will have different model configurations listed
as well as how well each performed. Select the best performing model, typically the
top entry of the list, and look for the json filename in that row which stores the best configuration.
With this file, we need to run k-fold cross validation to determine how well the model
actually performs.

Perform K-fold cross validation on the model for a more robust comparison of your model against other papers and models:

 - `cornell_grasp_train_classification.py` configuration file to train a single classification model completely.
     - Can do train/test/val splits
     - Can do k-fold cross validation
     - Be sure to manually edit the file with the json file specifying the hyperparameters for the model you wish to load, number of epochs, etc.

Here is one example configuration:

```
        FLAGS.load_hyperparams = ('hyperparams/classification/2018-05-03-17-02-01_inception_resnet_v2_classifier_model-'
                                  '_img_inception_resnet_v2_vec_dense_trunk_vgg_conv_block-dataset_cornell_grasping-grasp_success_hyperparams.json')
```

Here is the command to actually run k-fold training:

```
export CUDA_VISIBLE_DEVICES="0" && python cornell_grasp_train_classification.py  --run_name 2018-04-08-21-04-19_s2c2hw4 --pipeline_stage k_fold
```

After it finishes running there should be a file created named `*summary.json` with your final results.