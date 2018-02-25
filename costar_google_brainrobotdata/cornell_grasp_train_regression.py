'''
Training a network on cornell grasping dataset for regression of grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import tensorflow as tf
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(_):
    hyperparams, kwargs = grasp_utilities.load_hyperparams_json(
        FLAGS.load_hyperparams, FLAGS.fine_tuning, FLAGS.fine_tuning_learning_rate)

    # Override some default flags for this configuration
    FLAGS.problem_type = 'grasp_regression'
    FLAGS.feature_combo = 'image_preprocessed'
    if FLAGS.load_hyperparams is None:
        FLAGS.load_hyperparams = '/home/ahundt/datasets/logs/hyperopt_logs_cornell/2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json'
    FLAGS.split_dataset = 'objectwise'
    FLAGS.epochs = 100
    cornell_grasp_train.run_training(hyperparams=hyperparams, **kwargs)
    if 'k_fold' in FLAGS.pipeline_stage:
        cornell_grasp_train.train_k_fold()
    else:
        cornell_grasp_train.run_training()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()