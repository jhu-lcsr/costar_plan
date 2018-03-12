'''
Training a network on cornell grasping dataset for classification of grasp command proposals.

In other words, this network tries to predict a grasp rectangle from an input image.

The rectangle width is how open a gripper is,
the height is the length of the line along which grasps will be successful.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import tensorflow as tf
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(_):
    problem_type = 'grasp_classification'
    # recommended for
    # - pixelwise classification
    # - classification of images centered and rotated to grasp proposals
    feature_combo = 'image_preprocessed_norm_sin2_cos2_width_3'

    # Recommended for single prediction if images are merely center cropped
    # and not rotated or translated to the proposed grasp positions:
    # feature_combo = 'image_preprocessed_norm_sin2_cos2_w_yx_5'

    # recommended for
    # - classification of images centered and rotated to grasp proposals
    # feature_combo = 'image_preprocessed_width_1'

    # NOT RECOMMENDED due to height being correlated with grasp_success:
    # feature_combo = 'image_preprocessed_sin2_cos2_height_width_4'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'center_on_gripper_grasp_box_and_rotate_upright'
    if FLAGS.load_hyperparams is None:
        # FLAGS.load_hyperparams = ('/home/ahundt/datasets/logs/hyperopt_logs_cornell/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success/'
        #                           '2018-02-23-09-35-21_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')
        FLAGS.load_hyperparams = ('/home/ahundt/.keras/datasets/logs/hyperopt_logs_cornell_classification/'
                                  '2018-03-03-07-14-59_-vgg_dense_model-dataset_cornell_grasping-grasp_success/'
                                  '2018-03-03-07-14-59_-vgg_dense_model-dataset_cornell_grasping-grasp_success_hyperparams.json')
    FLAGS.epochs = 40
    FLAGS.fine_tuning_epochs = 10
    print('Classification Training on grasp_success is about to begin. '
          'This mode overrides some command line parameters so to change them '
          'you will need to modify cornell_grasp_train_classification.py directly.')

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
