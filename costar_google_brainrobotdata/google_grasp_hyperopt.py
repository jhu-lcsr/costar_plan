
from tensorflow.python.platform import flags

import hyperopt
import grasp_train
import grasp_utilities

FLAGS = flags.FLAGS


def main(_):
    FLAGS.grasp_success_label = 'move_to_grasp/time_ordered/grasp_success'
    FLAGS.problem_type = 'grasp_classification'
    FLAGS.grasp_sequence_motion_command_feature = 'move_to_grasp/time_ordered/reached_pose/transforms/endeffector_current_T_endeffector_final/vec_sin_cos_5'
    FLAGS.loss = 'binary_crossentropy'
    FLAGS.metric = 'binary_accuracy'
    FLAGS.num_validation = 1
    FLAGS.num_test = 1
    FLAGS.epochs = 1
    FLAGS.fine_tuning_epochs = 0
    print('Overriding some flags, edit google_grasp_hyperopt.py directly to change them.' +
          ' num_validation: ' + str(FLAGS.num_validation) +
          ' num_test: ' + str(FLAGS.num_test) +
          ' epochs: ' + str(FLAGS.epochs) +
          ' fine_tuning_epochs: ' + str(FLAGS.fine_tuning_epochs) +
          ' problem_type:' + str(FLAGS.problem_type))
    run_name = FLAGS.run_name
    log_dir = FLAGS.log_dir
    run_name = grasp_utilities.timeStamped(run_name)
    run_training_fn = grasp_train.run_training
    param_to_optimize = 'val_acc'
    batch_size = 4
    min_top_block_filter_multiplier = 6
    feature_combo_name = None

    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
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
