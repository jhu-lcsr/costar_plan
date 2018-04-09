#!/usr/local/bin/python
"""
Rank the results of hyperparameter optimization.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

"""

import os
import six
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app
import pandas
import grasp_utilities

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
    'log_dir',
    './hyperopt_logs_cornell_classification/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
)

flags.DEFINE_string(
    'glob_csv',
    '*/*.csv',
    'File path to glob for collecting hyperopt results.'
)


flags.DEFINE_string(
    'sort_by',
    'val_binary_accuracy',
    'variable name string to sort results by'
)

flags.DEFINE_boolean(
    'ascending',
    False,
    'Sort in ascending (1 to 100) or descending (100 to 1) order.'
)

flags.DEFINE_string(
    'save_csv',
    'hyperopt_rank.csv',
    'Where to save the sorted output csv file with the results'
)

flags.DEFINE_string(
    'save_dir',
    None,
    'Where to save the csv, defaults to log_dir'
)

flags.DEFINE_boolean(
    'print_results',
    False,
    'Print the results'
)

flags.DEFINE_boolean(
    'load_hyperparams',
    True,
    """load the hyperparameter include them in the results file.
       If something breaks try making this False.
    """
)

flags.DEFINE_string(
    'glob_hyperparams',
    '*hyperp*.json',
    'Hyperparams json filename strings to match in the directory of the corresponding csv file.'
)

FLAGS = flags.FLAGS


def main(_):
    csv_files = gfile.Glob(os.path.join(os.path.expanduser(FLAGS.log_dir), FLAGS.glob_csv))
    dataframe_list = []
    progress = tqdm(csv_files)
    for csv_file in progress:
        # progress.write('reading: ' + str(csv_file))
        try:
            dataframe = pandas.read_csv(csv_file, index_col=None, header=0)
            # add a filename column for this csv file's name
            dataframe['basename'] = os.path.basename(csv_file)
            dataframe['csv_filename'] = csv_file
            csv_dir = os.path.dirname(csv_file)
            hyperparam_filename = gfile.Glob(os.path.join(csv_dir, FLAGS.glob_hyperparams))
            if len(hyperparam_filename) > 1:
                progress.write('Unexpectedly got more than hyperparam file match, '
                                'only keeping the first one: ' + str(hyperparam_filename))
            hyperparam_filename = hyperparam_filename[0]
            dataframe['hyperparameters_filename'] = hyperparam_filename
            if FLAGS.load_hyperparams and len(hyperparam_filename) > 0:
                hyperparams = grasp_utilities.load_hyperparams_json(hyperparam_filename)
                for key, val in six.iteritems(hyperparams):
                    dataframe[key] = val
            dataframe_list.append(dataframe)
        except pandas.io.common.EmptyDataError as exception:
            # Ignore empty files, it just means hyperopt got killed early
            pass

    results_df = pandas.DataFrame()
    results_df = pandas.concat(dataframe_list)
    results_df = results_df.sort_values(FLAGS.sort_by, ascending=FLAGS.ascending)
    if FLAGS.print_results:
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
            print(results_df)
    if FLAGS.save_dir is None:
        FLAGS.save_dir = FLAGS.log_dir
    output_filename = os.path.join(FLAGS.save_dir, FLAGS.save_csv)
    results_df.to_csv(output_filename)
    print('Processing complete. Results saved to file: ' + str(output_filename))

if __name__ == '__main__':
    app.run(main=main)
