#!/usr/local/bin/python
'''
Training a network on cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code Based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import errno
import sys
import json
import csv
import argparse
import os.path
import glob
import datetime
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
import cornell_grasp_dataset_reader
import time
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


from keras import backend as K
import keras
import keras_contrib
import keras_tqdm
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import model_from_json
from grasp_model import concat_images_with_tiled_vector_layer
from grasp_model import top_block
from grasp_model import create_tree_roots
from grasp_model import choose_hypertree_model
from cornell_grasp_dataset_reader import parse_and_preprocess
from callbacks import EvaluateInputGenerator

import grasp_loss as grasp_loss


flags.DEFINE_float(
    'learning_rate',
    0.0341,
    'Initial learning rate.'
)
flags.DEFINE_float(
    'fine_tuning_learning_rate',
    0.006,
    'Initial learning rate.'
)
flags.DEFINE_integer(
    'epochs',
    20,
    'Number of epochs to run trainer.'
)
flags.DEFINE_integer(
    'batch_size',
    8,
    'Batch size.'
)
flags.DEFINE_string(
    'log_dir',
    './logs_cornell/',
    'Directory for tensorboard, model layout, model weight, csv, and hyperparam files'
)
flags.DEFINE_string(
    'model_path',
    '/tmp/tf/model.ckpt',
    'Variables for the model.'
)
flags.DEFINE_string(
    'train_or_validation',
    'validation',
    'Train or evaluate the dataset'
)
flags.DEFINE_string(
    'run_name',
    '',
    'A string that will become part of the logged directories and filenames.'
)
flags.DEFINE_integer(
    'num_splits',
    '10',
    'Total number of splits, which are equal in term of either imagewise or objectwise split'
)
flags.DEFINE_integer(
    'num_train',
    '8',
    'num of fold used for training, must be less than flags.train_splits'
)
flags.DEFINE_integer(
    'num_validation',
    '1',
    'num of fold used for validation, must be less than flags.train_splits'
)
flags.DEFINE_integer(
    'num_test',
    '1',
    'num of fold used for test, must be less than flags.train_splits'
)
flags.DEFINE_string('load_weights', '',
                    """Load and continue training the specified file containing model weights.""")
flags.DEFINE_string('pipeline_stage', 'train_test',
                    """Choose to "train", "test", "train_test", or "kfold" with the grasp_dataset
                       data for training and grasp_dataset_test for testing.""")
flags.DEFINE_string(
    'kfold_split_type', 'objectwise',
    """Options are imagewise and objectwise, this is the type of split chosen when the tfrecords were generated.""")
flags.DEFINE_string('tfrecord_filename_base', 'cornell-grasping-dataset', 'base of the filename used for the dataset tfrecords and csv files')

FLAGS = flags.FLAGS

# TODO(ahundt) put these utility functions in utils somewhere


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class PrintLogsCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('\nlogs:', logs)


def run_training(
        learning_rate=None,
        batch_size=None,
        num_gpus=1,
        top='classification',
        epochs=None,
        preprocessing_mode=None,
        save_model=True,
        train_data=None,
        validation_data=None,
        train_size=None,
        val_size=None,
        test_size=None,
        save_splits_weights='',
        feature_combo_name='image_preprocessed_sin2_cos2_width_3',
        problem_name='grasp_classification',
        image_model_name='vgg',
        optimizer_name='sgd',
        log_dir=None,
        hyperparams=None,
        load_weights=None,
        pipeline=None,
        **kwargs):
    """

    top: options are 'segmentation' and 'classification'.
    problem_name: options are 'grasp_regression', 'grasp_classification',
        'pixelwise_grasp_regression', 'pixelwise_grasp_classification',
        'image_center_grasp_regression'. Image center grasp regression is
        a pretraining step for pixel
        Make sure this is properly coordinated with 'top' param.
    feature_combo_name: The name for the combination of input features being utilized.
        Options include 'image_preprocessed', image_preprocessed_width_1,
        image_preprocessed_sin2_cos2_width_3
        'grasp_regression', image_center_grasp_regression.
        See choose_features_and_metrics() for details.
    hyperparams: a dictionary of hyperparameter selections made for this training run.
       If provided these values will simply be dumped to a file and
       not utilized in any other way.
    """
    if epochs is None:
        epochs = FLAGS.epochs
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if learning_rate is None:
        learning_rate = FLAGS.learning_rate
    if log_dir is None:
        log_dir = FLAGS.log_dir
    if load_weights is None:
        load_weights = FLAGS.load_weights
    if pipeline is None:
        pipeline = FLAGS.pipeline_stage

    [image_shapes, vector_shapes, data_features, model_name,
     monitor_loss_name, label_features, monitor_metric_name,
     loss, metrics, classes] = choose_features_and_metrics(feature_combo_name, problem_name)

    # see parse_and_preprocess() for the creation of these features
    model_name = image_model_name + model_name

    # If loading pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    preprocessing_mode = choose_preprocessing_mode(preprocessing_mode, image_model_name)

    # choose hypertree_model with inputs [image], [sin_theta, cos_theta]
    model = choose_hypertree_model(
        image_shapes=image_shapes,
        vector_shapes=vector_shapes,
        top=top,
        classes=classes,
        image_model_name=image_model_name,
        **kwargs)

    if(load_weights is not None and load_weights != ''):
        if os.path.isfile(load_weights):
            model.load_weights(load_weights)
        else:
            print('Could not load weights {}, '
                  'the file does not exist, '
                  'starting fresh....'.format(load_weights))
    print(monitor_loss_name)
    # TODO(ahundt) add a loss that changes size with how open the gripper is
    # loss = grasp_loss.segmentation_gaussian_measurement

    dataset_names_str = 'cornell_grasping'
    run_name = timeStamped(save_splits_weights + '-' + model_name + '-dataset_' + dataset_names_str + '-' + label_features[0])
    callbacks = []

    callbacks, optimizer = chooseOptimizer(optimizer_name, learning_rate, callbacks, monitor_loss_name)

    log_dir = os.path.join(log_dir, run_name)
    log_dir_run_name = os.path.join(log_dir, run_name)
    csv_logger = CSVLogger(log_dir_run_name + run_name + '.csv')
    callbacks = callbacks + [csv_logger]
    callbacks += [PrintLogsCallback()]
    print('Writing logs for models, accuracy and tensorboard in ' + log_dir)
    mkdir_p(log_dir)

    # Save the hyperparams to a json string so it is human readable
    if hyperparams is not None:
        with open(log_dir_run_name + '_hyperparams.json', 'w') as fp:
            json.dump(hyperparams, fp)

    # Save the current model to a json string so it is human readable
    with open(log_dir_run_name + '_model.json', 'w') as fp:
        fp.write(model.to_json())

    checkpoint = keras.callbacks.ModelCheckpoint(
        log_dir_run_name + '-epoch-{epoch:03d}-' +
        monitor_loss_name + '-{' + monitor_loss_name + ':.3f}-' +
        monitor_metric_name + '-{' + monitor_metric_name + ':.3f}.h5',
        save_best_only=True, verbose=1, monitor=monitor_metric_name)

    callbacks = callbacks + [checkpoint]
    # An additional useful param is write_batch_performance:
    #  https://github.com/keras-team/keras/pull/7617
    #  write_batch_performance=True)
    progress_tracker = TensorBoard(log_dir=log_dir, write_graph=True,
                                   write_grads=False, write_images=False,
                                   histogram_freq=0, batch_size=batch_size)
                                   # histogram_freq=0, batch_size=batch_size,
                                   # write_batch_performance=True)
    callbacks = callbacks + [progress_tracker]

    # make sure the TQDM callback is always the final one
    callbacks += [keras_tqdm.TQDMCallback()]

    if num_gpus > 1:
        parallel_model = keras.utils.multi_gpu_model(model, num_gpus)
    else:
        parallel_model = model

    parallel_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    train_data, train_steps, validation_data, validation_steps, test_data, test_steps = load_dataset(
        label_features=label_features, data_features=data_features, batch_size=batch_size,
        train_data=train_data, validation_data=validation_data, preprocessing_mode=preprocessing_mode)

    # Get the validation dataset in one big numpy array for validation
    # This lets us take advantage of tensorboard visualization
    if 'train' in pipeline:
        if 'test' in pipeline:
            # we need this callback to be at the beginning!
            callbacks = [EvaluateInputGenerator(generator=test_data,
                                                steps=test_steps,
                                                metrics_prefix='test',
                                                verbose=0)] + callbacks

        # print('calling model.fit_generator()')
        history = parallel_model.fit_generator(
            train_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=0)

    elif pipeline == 'test':
        history = parallel_model.evaluate_generator(generator=test_data, steps=test_steps)

    model.save_weights(log_dir_run_name + '_model_weights.h5')

    return history


def choose_preprocessing_mode(preprocessing_mode, image_model_name):
    """ Choose preprocessing for specific pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    """
    if preprocessing_mode is None:
        if 'densenet' in image_model_name:
            preprocessing_mode = 'torch'
        elif 'nasnet' in image_model_name:
            preprocessing_mode = 'tf'
        elif 'vgg' in image_model_name:
            preprocessing_mode = 'tf'
        elif 'resnet' in image_model_name:
            preprocessing_mode = 'caffe'
        else:
            raise ValueError('You need to explicitly set the preprocessing mode to '
                             'torch, tf, or caffe for these weights')
    return preprocessing_mode


def chooseOptimizer(optimizer_name, learning_rate, callbacks, monitor_loss_name):
    if optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate * 1.0)
        callbacks = callbacks + [
            # Reduce the learning rate if training plateaus.
            keras.callbacks.ReduceLROnPlateau(patience=12, verbose=1, factor=0.5, monitor=monitor_loss_name)
        ]
    elif optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam()
    else:
        raise ValueError('Unsupported optimizer ' + str(optimizer_name) +
                         'try adam or sgd.')
    return callbacks, optimizer


def train_k_fold(num_fold=None, split_type='imagewise', csv_path='-k-fold-stat.csv'):
    """ Do K_Fold training

        num_fold: total number of fold.
        split_type: str, either 'imagewise' or 'objectwise', should be consistent with
        splits type desired when doing actual splits.
    """
    cur_csv_path = os.path.join(FLAGS.data_dir, split_type + csv_path)
    csv_reader = csv.DictReader(open(cur_csv_path))
    unique_image_num = []
    for row in csv_reader:
        unique_image_num.append(int(row[' unique_image']))
    if num_fold is None:
        num_fold = FLAGS.num_splits
    val_filenames = []
    train_filenames = []
    train_id = ''
    val_size = 0
    train_size = 0
    for i in range(num_fold):
        val_filenames = [os.path.join(FLAGS.data_dir,
                         'cornell-grasping-dataset' + split_type + '-fold-' + str(i) + '.tfrecord')]
        val_size = unique_image_num[i]
        for j in range(num_fold):
            if j == i:
                continue
            train_id += str(j)
            train_filenames += [os.path.join(FLAGS.data_dir,
                                'cornell-grasping-dataset' + split_type + '-fold-' + str(j) + '.tfrecord')]
            train_size += unique_image_num[j]
        save_splits_weights = split_type + '_train_on_' + train_id + '_val_on_' + str(i)
        run_training(train_files=train_filenames, validation_files=val_filenames, save_splits_weights=save_splits_weights,
                     train_size=train_size, val_size=val_size)
        train_size = 0

    return


def choose_features_and_metrics(feature_combo_name, problem_name):
    """ Choose the features to load from the dataset and losses to use during training
    """
    image_shapes = [(FLAGS.resize_height, FLAGS.resize_width, 3)]
    classes = 1

    # TODO(ahundt) get input dimensions automatically, based on configured params
    if feature_combo_name == 'image/preprocessed' or feature_combo_name == 'image_preprocessed':
        data_features = ['image/preprocessed']
        vector_shapes = None
    elif feature_combo_name == 'image_preprocessed_sin2_cos2_height_width_4':
        # don't use this one as an input! height appears highly correlated with grasp_success.
        data_features = ['image/preprocessed', 'preprocessed_sin2_cos2_height_width_4']
        vector_shapes = [(4,)]
    elif feature_combo_name == 'image_preprocessed_sin_cos_width_3':
        data_features = ['image/preprocessed', 'preprocessed_sin_cos_width_3']
        vector_shapes = [(3,)]
    elif feature_combo_name == 'image_preprocessed_sin2_cos2_width_3':
        data_features = ['image/preprocessed', 'preprocessed_sin2_cos2_width_3']
        vector_shapes = [(3,)]
    elif feature_combo_name == 'image_preprocessed_width_1':
        data_features = ['image/preprocessed', 'bbox/width']
        vector_shapes = [(1,)]
    elif feature_combo_name == 'preprocessed':
        data_features = ['image/preprocessed', 'bbox/preprocessed/cy_cx_normalized_2',
                         'bbox/preprocessed/sin_cos_2', 'bbox/preprocessed/logarithm_height_width_2']
        vector_shapes = [(2,), (2,), (2,)]
    elif feature_combo_name == 'raw':
        data_features = ['image/decoded', 'sin_cos_height_width_4']
        vector_shapes = [(4,)]
    else:
        raise ValueError('Selected feature ' + str(feature_combo_name) + ' does not exist. '
                         'feature selection options are image_preprocessed_sin_cos_height_width_4, '
                         'image_preprocessed_sin_cos_height_3, image_preprocessed_height_1,'
                         'preprocessed, and raw')

    if problem_name == 'segmentation' or problem_name == 'grasp_segmentation' or problem_name == 'pixelwise_grasp_classification':
        label_features = ['grasp_success_yx_3']
        monitor_loss_name = 'segmentation_gaussian_binary_crossentropy'
        monitor_metric_name = 'val_segmentation_single_pixel_binary_accuracy'
        loss = grasp_loss.segmentation_gaussian_binary_crossentropy
        metrics = [grasp_loss.segmentation_single_pixel_binary_accuracy, grasp_loss.mean_pred]
        model_name = '_dilated_model'
    elif problem_name == 'classification' or problem_name == 'grasp_classification':
        label_features = ['grasp_success']
        monitor_metric_name = 'val_binary_accuracy'
        monitor_loss_name = 'val_loss'
        metrics = ['binary_accuracy', grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.binary_crossentropy
        model_name = '_dense_model'
    elif problem_name == 'grasp_regression':
        label_features = ['grasp_success_sin2_cos2_hw_norm_yx_7']
        monitor_metric_name = 'val_mean_squared_error'
        monitor_loss_name = 'val_loss'
        metrics = [keras.losses.mean_squared_error, grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.mean_squared_error
        model_name = '_regression_model'
        classes = 7
    elif problem_name == 'image_center_grasp_regression':
        label_features = ['grasp_success_sin2_cos2_hw_5']
        monitor_metric_name = 'val_mean_squared_error'
        monitor_loss_name = 'val_loss'
        metrics = [keras.losses.mean_squared_error, grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.mean_squared_error
        model_name = '_center_regression_model'
        classes = 5
    else:
        raise ValueError('Selected problem_name ' + str(problem_name) + ' does not exist. '
                         'feature selection options are segmentation and classification')
    return image_shapes, vector_shapes, data_features, model_name, monitor_loss_name, label_features, monitor_metric_name, loss, metrics, classes


def load_dataset(label_features, data_features,
                 batch_size, val_batch_size=1, test_batch_size=1,
                 train_data=None, validation_data=None, test_data=None, in_memory_validation=False,
                 preprocessing_mode='tf'):
    """ Load the cornell grasping dataset from the file if it isn't already available.
    """
    train_filenames, train_size, val_filenames, val_size, test_filenames, test_size = epoch_params()
    train_steps, val_steps, test_steps = epoch_params_for_splits(
        train_batch=batch_size, val_batch=val_batch_size, test_batch=test_batch_size,
        samples_train=train_size, samples_val=val_size, samples_test=test_size)

    if in_memory_validation:
        val_batch_size = val_size

    if validation_data is None:
        validation_data = cornell_grasp_dataset_reader.yield_record(
            val_filenames, label_features, data_features,
            is_training=False, batch_size=val_batch_size,
            parse_example_proto_fn=parse_and_preprocess,
            preprocessing_mode=preprocessing_mode)

    if in_memory_validation:
        print('loading validation data directly into memory, if you run out set in_memory_validation to False')
        validation_data = next(validation_data)

    if train_data is None:
        train_data = cornell_grasp_dataset_reader.yield_record(
            train_filenames, label_features, data_features,
            batch_size=batch_size,
            parse_example_proto_fn=parse_and_preprocess,
            preprocessing_mode=preprocessing_mode)

    if test_data is None:
        test_data = cornell_grasp_dataset_reader.yield_record(
            test_filenames, label_features, data_features,
            batch_size=test_batch_size,
            parse_example_proto_fn=parse_and_preprocess,
            preprocessing_mode=preprocessing_mode)

    return train_data, train_steps, validation_data, val_steps, test_data, test_steps


def epoch_params(train_splits=None, val_splits=None, test_splits=None, split_type=None,
                 csv_path='-k-fold-stat.csv', data_dir=None, tfrecord_filename_base=None):
    """ Determine the number of steps to train, validate, and test

    TODO(ahundt) WARNING: THE NUMBER OF TRAIN/VAL STEPS VARIES EVERY TIME THE DATASET IS CONVERTED, AUTOMATE SETTING THOSE NUMBERS
    """
    if data_dir is None:
        data_dir = FLAGS.data_dir

    if split_type is None:
        split_type = FLAGS.kfold_split_type

    if tfrecord_filename_base is None:
        tfrecord_filename_base = FLAGS.tfrecord_filename_base
    # must be sure that train_splits + val_splits + test_filenames = flags.num_splits
    cur_csv_path = os.path.join(data_dir, tfrecord_filename_base + '-' + split_type + csv_path)
    with open(cur_csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        unique_image_num = []
        for row in csv_reader:
            unique_image_num.append(int(row[' num_total_grasp']))

        if train_splits is None:
            train_splits = FLAGS.num_train
        if val_splits is None:
            val_splits = FLAGS.num_validation
        if test_splits is None:
            test_splits = FLAGS.num_test

        train_filenames = []
        val_filenames = []
        test_filenames = []

        train_size = 0
        val_size = 0
        test_size = 0
        for i in range(train_splits):
            train_size += unique_image_num[i]
            train_filenames += [os.path.join(data_dir,
                                tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        for i in range(train_splits, train_splits + val_splits):
            val_size += unique_image_num[i]
            val_filenames += [os.path.join(data_dir,
                              tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        for i in range(train_splits + val_splits, train_splits + val_splits + test_splits):
            test_size += unique_image_num[i]
            test_filenames += [os.path.join(data_dir,
                               tfrecord_filename_base + '-' + split_type + '-fold-' + str(i) + '.tfrecord')]

        return train_filenames, train_size, val_filenames, val_size, test_filenames, test_size

        # samples_in_training_dataset = 6402
        # samples_in_val_dataset = 1617
        # val_batch_size = 11
        # steps_in_val_dataset, divides_evenly = np.divmod(samples_in_val_dataset, val_batch_size)
        # assert divides_evenly == 0
        # steps_per_epoch_train = np.ceil(float(samples_in_training_dataset) / float(batch_size))
        # return samples_in_val_dataset, steps_per_epoch_train, steps_in_val_dataset, val_batch_size


def epoch_params_for_splits(train_batch=None, samples_train=None,
                            val_batch=None, samples_val=None,
                            test_batch=None, samples_test=None):

    returns = []
    if samples_train is not None and train_batch is not None:
        # for training, just do a little more than once through the dataset if needed
        steps_train = int(np.ceil(float(samples_train) / float(train_batch)))
        returns.append(steps_train)
    if samples_val is not None and val_batch is not None:
        steps_in_val_dataset, divides_evenly = np.divmod(samples_val, val_batch)
        # If this fails you need to fix the batch size so it divides evenly!
        assert divides_evenly == 0
        returns.append(steps_in_val_dataset)
    if samples_test is not None and test_batch is not None:
        steps_in_test_dataset, divides_evenly = np.divmod(samples_test, test_batch)
        # If this fails you need to fix the batch size so it divides evenly!
        assert divides_evenly == 0
        returns.append(steps_in_test_dataset)

    return returns


def bboxes_to_grasps(bboxes):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    box = tf.unstack(bboxes, axis=1)
    x = (box[0] + (box[4] - box[0])/2) * 0.35
    y = (box[1] + (box[5] - box[1])/2) * 0.47
    tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35
    h = tf.sqrt(tf.pow((box[2] -box[0])*0.35, 2) + tf.pow((box[3] -box[1])*0.47, 2))
    w = tf.sqrt(tf.pow((box[6] -box[0])*0.35, 2) + tf.pow((box[7] -box[1])*0.47, 2))
    return x, y, tan, h, w


def grasp_to_bbox(x, y, tan, h, w):
    theta = tf.atan(tan)
    edge1 = (x -w/2*tf.cos(theta) +h/2*tf.sin(theta), y -w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge2 = (x +w/2*tf.cos(theta) +h/2*tf.sin(theta), y +w/2*tf.sin(theta) -h/2*tf.cos(theta))
    edge3 = (x +w/2*tf.cos(theta) -h/2*tf.sin(theta), y +w/2*tf.sin(theta) +h/2*tf.cos(theta))
    edge4 = (x -w/2*tf.cos(theta) -h/2*tf.sin(theta), y -w/2*tf.sin(theta) +h/2*tf.cos(theta))
    return [edge1, edge2, edge3, edge4]


def old_run_training():
    print(FLAGS.train_or_validation)
    if FLAGS.train_or_validation == 'train':
        print('distorted_inputs')
        data_files_ = TRAIN_FILE
        features = cornell_grasp_dataset_reader.distorted_inputs(
                  [data_files_], FLAGS.epochs, batch_size=FLAGS.batch_size)
    else:
        print('inputs')
        data_files_ = VALIDATE_FILE
        features = cornell_grasp_dataset_reader.inputs([data_files_])

    # loss, x_hat, tan_hat, h_hat, w_hat, y_hat = old_loss(tan, x, y, h, w)
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = keras.backend.get_session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    d={}
    l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
    for i in l:
        d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    saver = tf.train.Saver(d)
    saver_g = tf.train.Saver(dg)
    #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
    saver_g.restore(sess, FLAGS.model_path)
    try:
        count = 0
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train
            if FLAGS.train_or_validation == 'train':
                _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, x, x_hat, tan, tan_hat, h, h_hat, w, w_hat])
                duration = time.time() - start_batch
                if step % 100 == 0:
                    print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n')%(step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
                if step % 1000 == 0:
                    saver_g.save(sess, FLAGS.model_path)
            else:
                bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat)
                bbox_value, bbox_model, tan_value, tan_model = sess.run([bboxes, bbox_hat, tan, tan_hat])
                bbox_value = np.reshape(bbox_value, -1)
                bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)]
                p1 = Polygon(bbox_value)
                p2 = Polygon(bbox_model)
                iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
                angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
                duration = time.time() -start_batch
                if angle_diff < 30. and iou >= 0.25:
                    count+=1
                    print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
            step +=1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.epochs, step, (time.time()-start_time)/60))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def old_loss(tan, x, y, h, w):
    from grasp_inf import inference
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) # list
    # tangent of 85 degree is 11
    tan_hat_confined = tf.minimum(11., tf.maximum(-11., tan_hat))
    tan_confined = tf.minimum(11., tf.maximum(-11., tan))
    # Loss function
    gamma = tf.constant(10.)
    loss = tf.reduce_sum(tf.pow(x_hat -x, 2) +tf.pow(y_hat -y, 2) + gamma*tf.pow(tan_hat_confined - tan_confined, 2) +tf.pow(h_hat -h, 2) +tf.pow(w_hat -w, 2))
    return loss, x_hat, tan_hat, h_hat, w_hat, y_hat


def main(_):
    run_training()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
