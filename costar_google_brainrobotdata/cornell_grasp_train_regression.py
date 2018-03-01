'''
Training a network on cornell grasping dataset for regression of grasping positions.

In other words, this network tries to predict a grasp rectangle from an input image.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import tensorflow as tf
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(_):
    problem_type = 'grasp_regression'
    feature_combo = 'image_preprocessed'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'image_contains_grasp_box_center'
    if FLAGS.load_hyperparams is None:
        FLAGS.load_hyperparams = '/home/ahundt/datasets/logs/hyperopt_logs_cornell/2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json'
    FLAGS.epochs = 60
    FLAGS.fine_tuning_epochs = 5
    print('Regression Training on Jaccard Distance is about to begin. '
          'It overrides some command line parameters so to change them '
          'you will need to modify cornell_grasp_train_regression.py directly.')

    hyperparams = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, FLAGS.fine_tuning, FLAGS.learning_rate,
        feature_combo_name=feature_combo)

    if 'k_fold' in FLAGS.pipeline_stage:
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='objectwise',
            **hyperparams)
        cornell_grasp_train.train_k_fold(
            problem_name=problem_type,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            split_type='imagewise',
            **hyperparams)
    else:
        cornell_grasp_train.run_training(
            problem_name=problem_type,
            feature_combo_name=feature_combo,
            hyperparams=hyperparams,
            **hyperparams)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
