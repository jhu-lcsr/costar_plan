'''
Training a network on cornell grasping dataset for regression of grasping positions.
This will go through sorted hyperopt results from best to worst and train each model,
ranking the best final results.

In other words, this network tries to predict a grasp rectangle from an input image.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

'''
import sys
import os
import traceback
import tensorflow as tf
import pandas
import json
import grasp_utilities
import cornell_grasp_train
from tensorflow.python.platform import flags

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


flags.DEFINE_string(
    'rank_csv',
    'hyperopt_logs_costar_grasp_regression/hyperopt_rank.csv',
    """Sorted csv ranking models on which to perform full runs after hyperparameter optimization.
    See cornell_hyperopt.py to perform hyperparameter optimization, and then hyperopt_rank.py to generate the ranking csv file."""
)


FLAGS = flags.FLAGS


def main(_):
    use_best_model = True
    problem_type = 'semantic_translation_regression'
    # problem_type = 'semantic_rotation_regression'
    # problem_type = 'semantic_grasp_regression'
    feature_combo = 'image_preprocessed'
    # Override some default flags for this configuration
    # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
    FLAGS.problem_type = problem_type
    FLAGS.feature_combo = feature_combo
    FLAGS.crop_to = 'image_contains_grasp_box_center'
    load_weights = None
    FLAGS.batch_size = 256
    optimizer_name = 'sgd'
    # FLAGS.crop_height = 480
    # FLAGS.crop_width = 640
    # FLAGS.resize_height = 480
    # FLAGS.resize_width = 640
    # print('Note: special overrides have been applied '
    #       'for an experiment. '
    #       'crop + resize width/height have been set to 640x480.')
    # FLAGS.log_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_plan/costar_google_brainrobotdata/hyperparams/'
    # FLAGS.data_dir = r'C:/Users/Varun/JHU/LAB/Projects/costar_block_stacking_dataset_v0.2/*success.h5f'

    FLAGS.data_dir = os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.2/*success.h5f')
    FLAGS.fine_tuning_epochs = 0
    FLAGS.epochs = 20
    print('Regression Training on costar block stacking is about to begin. '
          'It overrides some command line parameters including '
          'training on mae loss so to change them '
          'you will need to modify cornell_grasp_train_regression.py directly.')

    dataset_name = 'costar_block_stacking'

    json_histories_path = os.path.join(FLAGS.log_dir, FLAGS.run_name + '_' + problem_type + '_histories.json')
    json_histories_summary_path = os.path.join(FLAGS.log_dir, FLAGS.run_name + '_' + problem_type + '_histories_summary.json')
    run_histories = {}
    history_dicts = {}

    dataframe = pandas.read_csv(FLAGS.rank_csv, index_col=None, header=0)
    row_progress = tqdm(dataframe.iterrows())
    for index, row in row_progress:
        history = None
        hyperparams_filename = row['hyperparameters_filename']

        hyperparams = grasp_utilities.load_hyperparams_json(
            hyperparams_filename, FLAGS.fine_tuning, FLAGS.learning_rate,
            feature_combo_name=feature_combo)

        row_progress.write('-' * 80)
        row_progress.write('Training with hyperparams at index ' + str(index) + ' from: ' + str(hyperparams_filename) + '\n\n' + str(hyperparams))
        row_progress.write('-' * 80)
        hyperparams['loss'] = 'mse'
        # save weights at checkpoints as the model's performance improves
        hyperparams['checkpoint'] = True
        hyperparams['learning_rate'] = 1.0

        try:
            history = cornell_grasp_train.run_training(
                problem_name=problem_type,
                # feature_combo_name=feature_combo,
                hyperparams=hyperparams,
                dataset_name=dataset_name,
                optimizer_name=optimizer_name,
                load_weights=load_weights,
                **hyperparams)

            run_histories[hyperparams_filename] = history
            history_dicts[hyperparams_filename] = history.history
            # save the histories so far, overwriting past updates
            with open(json_histories_path, 'w') as fp:
                # save out all kfold params so they can be reloaded in the future
                json.dump(history_dicts, fp, cls=grasp_utilities.NumpyEncoder)

            results = grasp_utilities.multi_run_histories_summary(
                run_histories,
                save_filename=json_histories_summary_path,
                description_prefix=problem_type + 'min_',
                results_prefix='ranked_regression_min_results',
                multi_history_metric='max'
            )
        except tf.errors.ResourceExhaustedError as exception:
            print('Hyperparams caused algorithm to run out of resources, '
                  'will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you might set more memory sensitive hyperparam ranges,'
                  'or add constraints to your hyperparam search so it does not choose'
                  'huge values for all the parameters at once'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except (ValueError, tf.errors.FailedPreconditionError, tf.errors.OpError) as exception:
            print('Hyperparams encountered a model that failed with an invalid combination of values, '
                  'we will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you will need to debug your model w.r.t. '
                  'the current hyperparam choice.'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except KeyboardInterrupt as e:
            print('Evaluation of this model canceled based on a user request. '
                  'We will continue to next stage and return infinity loss for the canceled model.')
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb

        # TODO(ahundt) consider shutting down dataset generators and clearing the session when there is an exception
        # https://github.com/tensorflow/tensorflow/issues/4735#issuecomment-363748412
        keras.backend.clear_session()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
