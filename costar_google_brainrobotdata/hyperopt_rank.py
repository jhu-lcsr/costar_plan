#!/usr/local/bin/python
"""
Rank the results of hyperparameter optimization.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

"""

import os
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.app import run
import pandas


flags.DEFINE_string(
    'log_dir',
    './hyperopt_logs_cornell_classification/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
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

FLAGS = flags.FLAGS


def main(_):
    csv_files = gfile.Glob(os.path.join(os.path.expanduser(FLAGS.log_dir), '*/*.csv'))
    dataframe_list = []
    for csv_file in csv_files:
        dataframe = pandas.read_csv(csv_file, index_col=None, header=0)
        # add a filename column for this csv file's name
        dataframe['filename'] = csv_file
        dataframe_list.append(dataframe)

    results_df = pandas.DataFrame()
    results_df.concat(dataframe_list)
    results_df.sort_values(FLAGS.sort_by, ascending=FLAGS.ascending)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df)
    if save_dir is None:
        save_dir = FLAGS.log_dir
    results_df.to_csv(save_dir)

if __name__ == '__main__':
    run(main=main)
