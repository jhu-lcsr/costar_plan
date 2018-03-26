
from tensorflow.python.platform import flags

import hyperopt
import cornell_grasp_train
import cornell_grasp_dataset_reader
import grasp_utilities

FLAGS = flags.FLAGS


def cornell_hyperoptions(problem_type, param_to_optimize):
    """ Set some hyperparams based on the problem type and parameter to optimize

    Deal with some string renames and modes in which the hyperparameter
    optimization algorithm can run such as grasp classification vs regression.
    """
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    batch_size = FLAGS.batch_size
    if problem_type == 'classification':
        FLAGS.problem_type = problem_type
        # note: this version will have some correlation with success,
        # but it will be OK to use to classify the output of regression
        feature_combo_name = 'image_preprocessed_norm_sin2_cos2_height_width_4'

        # recommended for
        # - pixelwise classification
        # - classification of images centered and rotated to grasp proposals
        # feature_combo = 'image_preprocessed_norm_sin2_cos2_width_3'

        # Another simpler option with less input data:
        # feature_combo_name = 'image_preprocessed_width_1'
        FLAGS.crop_to = 'center_on_gripper_grasp_box_and_rotate_upright'
        if param_to_optimize == 'val_acc':
            param_to_optimize = 'val_binary_accuracy'
        min_top_block_filter_multiplier = 6
    elif problem_type == 'grasp_regression' or 'regression':
        feature_combo_name = 'image_preprocessed'
        # Override some default flags for this configuration
        # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
        FLAGS.problem_type = problem_type
        FLAGS.feature_combo = feature_combo_name
        FLAGS.crop_to = 'image_contains_grasp_box_center'
        if param_to_optimize == 'val_acc':
            param_to_optimize = 'val_grasp_jaccard'
        min_top_block_filter_multiplier = 8
    return feature_combo_name, min_top_block_filter_multiplier, batch_size, param_to_optimize


def main(_):

    FLAGS.problem_type = 'grasp_regression'
    FLAGS.num_validation = 1
    FLAGS.num_test = 1
    FLAGS.epochs = 5
    FLAGS.fine_tuning_epochs = 0
    print('Overriding some flags, edit cornell_hyperopt.py directly to change them.' +
          ' num_validation: ' + str(FLAGS.num_validation) +
          ' num_test: ' + str(FLAGS.num_test) +
          ' epochs: ' + str(FLAGS.epochs) +
          ' fine_tuning_epochs: ' + str(FLAGS.fine_tuning_epochs) +
          ' problem_type:' + str(FLAGS.problem_type))
    run_name = FLAGS.run_name
    log_dir = FLAGS.log_dir
    run_name = grasp_utilities.timeStamped(run_name)
    run_training_fn = cornell_grasp_train.run_training
    problem_type = FLAGS.problem_type
    param_to_optimize = 'val_acc'

    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    feature_combo_name, min_top_block_filter_multiplier, batch_size, param_to_optimize = cornell_hyperoptions(problem_type, param_to_optimize)
    hyperopt.optimize(
        run_training_fn=run_training_fn,
        feature_combo_name=feature_combo_name,
        problem_type=FLAGS.problem_type,
        run_name=run_name,
        log_dir=log_dir,
        min_top_block_filter_multiplier=min_top_block_filter_multiplier,
        batch_size=batch_size,
        param_to_optimize=param_to_optimize)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
