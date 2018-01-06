"""Code for training models on the google brain robotics grasping dataset.

https://sites.google.com/site/brainrobotdata/home/grasping-dataset

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""
import os
import sys
import datetime
import traceback
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from tensorflow.python.platform import flags

import grasp_dataset
import grasp_model
import grasp_loss

from tqdm import tqdm  # progress bars https://github.com/tqdm/tqdm
# from keras_tqdm import TQDMCallback  # Keras tqdm progress bars https://github.com/bstriner/keras-tqdm

try:
    import horovod.keras as hvd
except ImportError:
    print('Horovod is not installed, see https://github.com/uber/horovod.'
          'Distributed training is disabled but single machine training '
          'should continue to work but without learning rate warmup.')
    hvd = None

flags.DEFINE_string('learning_rate_decay_algorithm', 'power_decay',
                    """Determines the algorithm by which learning rate decays,
                       options are power_decay, exp_decay, adam and progressive_drops.
                       Only applies with optimizer flag is SGD""")
flags.DEFINE_string('grasp_model', 'grasp_model_single',
                    """Choose the model definition to run, options are:
                       grasp_model_levine_2016, grasp_model, grasp_model_resnet, grasp_model_segmentation""")
flags.DEFINE_string('save_weights', 'grasp_model_weights',
                    """Save a file with the trained model weights.""")
flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
                    """Load and continue training the specified file containing model weights.""")
flags.DEFINE_integer('epochs', 300,
                     """Epochs of training""")
flags.DEFINE_string('grasp_dataset_eval', '097',
                    """Filter the subset of 1TB Grasp datasets to evaluate.
                    097 by default. It is important to ensure that this selection
                    is completely different from the selected training datasets
                    with no overlap, otherwise your results won't be valid!
                    See https://sites.google.com/site/brainrobotdata/home
                    for a full listing.""")
flags.DEFINE_string('pipeline_stage', 'train_eval',
                    """Choose to "train", "eval", or "train_eval" with the grasp_dataset
                       data for training and grasp_dataset_eval for evaluation.""")
flags.DEFINE_float('learning_rate_scheduler_power_decay_rate', 1.5,
                   """Determines how fast the learning rate drops at each epoch.
                      lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
                      Training from scratch within an initial learning rate of 0.1 might find a
                         power decay value of 2 to be useful.
                      Fine tuning with an initial learning rate of 0.001 may consder 1.5 power decay.""")
flags.DEFINE_float('grasp_learning_rate', 0.01,
                   """Determines the initial learning rate""")
flags.DEFINE_integer('eval_batch_size', 1, 'batch size per compute device')
flags.DEFINE_integer('densenet_growth_rate', 12,
                     """DenseNet and DenseNetFCN parameter growth rate""")
flags.DEFINE_integer('densenet_depth', 40,
                     """DenseNet total number of layers, aka depth""")
flags.DEFINE_integer('densenet_dense_blocks', 3,
                     """The number of dense blocks in the model.""")
flags.DEFINE_float('densenet_reduction', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter.""")
flags.DEFINE_float('densenet_reduction_after_pretrained', 0.5,
                   """DenseNet and DenseNetFCN reduction aka compression parameter,
                      applied to the second DenseNet component after pretrained imagenet models.""")
flags.DEFINE_float('dropout_rate', 0.2,
                   """Dropout rate for the model during training.""")
flags.DEFINE_string('eval_results_file', 'grasp_model_eval.txt',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_string('device', '/gpu:0',
                    """Save a file with results of model evaluation.""")
flags.DEFINE_bool('tf_allow_memory_growth', True,
                  """False if memory usage will be allocated all in advance
                     or True if it should grow as needed. Allocating all in
                     advance may reduce fragmentation.""")
flags.DEFINE_string('learning_rate_scheduler', 'learning_rate_scheduler',
                    """Options are None and learning_rate_scheduler,
                       turning this on activates the scheduler which follows
                       a power decay path for the learning rate over time.
                       This is most useful with SGD, currently disabled with Adam.""")
flags.DEFINE_string('optimizer', 'SGD', """Options are Adam and SGD.""")
flags.DEFINE_string('progress_tracker', None,
                    """Utility to follow training progress, options are tensorboard and None.""")
flags.DEFINE_string('loss', 'segmentation_single_pixel_binary_crossentropy',
                    """Options are binary_crossentropy and segmentation_single_pixel_binary_crossentropy.""")
flags.DEFINE_string('metric', 'segmentation_single_pixel_binary_accuracy',
                    """Options are accuracy, binary_accuracy and segmentation_single_pixel_binary_accuracy.""")

flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class GraspTrain(object):

    def train(self, dataset=FLAGS.grasp_datasets_train,
              grasp_datasets_batch_algorithm=FLAGS.grasp_datasets_batch_algorithm,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              load_weights=FLAGS.load_weights,
              save_weights=FLAGS.save_weights,
              make_model_fn=grasp_model.grasp_model,
              imagenet_preprocessing=FLAGS.imagenet_preprocessing,
              grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
              grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
              random_crop=FLAGS.random_crop,
              resize=FLAGS.resize,
              resize_height=FLAGS.resize_height,
              resize_width=FLAGS.resize_width,
              learning_rate_decay_algorithm=FLAGS.learning_rate_decay_algorithm,
              learning_rate=FLAGS.grasp_learning_rate,
              learning_power_decay_rate=FLAGS.learning_rate_scheduler_power_decay_rate,
              dropout_rate=FLAGS.dropout_rate,
              model_name=FLAGS.grasp_model,
              loss=FLAGS.loss,
              metric=FLAGS.metric):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """

        if hvd is not None:
            # Initialize Horovod.
            hvd.init()

        # Pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

        datasets = dataset.split(',')
        (pregrasp_op_batch,
         grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         grasp_success_op_batch,
         steps_per_epoch) = grasp_dataset.get_multi_dataset_training_tensors(
             datasets,
             batch_size,
             grasp_datasets_batch_algorithm,
             imagenet_preprocessing,
             random_crop,
             resize,
             grasp_sequence_min_time_step,
             grasp_sequence_max_time_step)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training

        weights_name = timeStamped(save_weights + '-' + model_name)

        # ###############learning rate scheduler####################
        # source: https://github.com/aurora95/Keras-FCN/blob/master/train.py
        # some quick lines to see what a power_decay schedule would do at each epoch:
        # import numpy as np
        # epochs = 100
        # learning_rate = 0.1
        # learning_power_decay_rate = 2
        # print([learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate) for epoch in np.arange(epochs)])

        def lr_scheduler(epoch, learning_rate=learning_rate,
                         mode=learning_rate_decay_algorithm,
                         epochs=epochs,
                         learning_power_decay_rate=learning_power_decay_rate):
            """if lr_dict.has_key(epoch):
                lr = lr_dict[epoch]
                print 'lr: %f' % lr
            """

            if mode is 'power_decay':
                # original lr scheduler
                lr = learning_rate * ((1 - float(epoch)/epochs) ** learning_power_decay_rate)
            if mode is 'exp_decay':
                # exponential decay
                lr = (float(learning_rate) ** float(learning_power_decay_rate)) ** float(epoch+1)
            # adam default lr
            if mode is 'adam':
                lr = 0.001

            if mode is 'progressive_drops':
                # drops as progression proceeds, good for sgd
                if epoch > 0.9 * epochs:
                    lr = 0.0001
                elif epoch > 0.75 * epochs:
                    lr = 0.001
                elif epoch > 0.5 * epochs:
                    lr = 0.01
                else:
                    lr = 0.1

            print('lr: %f' % lr)
            return lr

        # TODO(ahundt) manage loss/accuracy names in a more principled way
        loss_name = 'loss'
        if 'segmentation_single_pixel_binary_crossentropy' in loss:
            loss = grasp_loss.segmentation_single_pixel_binary_crossentropy
            loss_name = 'segmentation_single_pixel_binary_crossentropy'

        metric_name = 'acc'
        if 'segmentation_single_pixel_binary_accuracy' in metric:
            metric_name = metric
            metric = grasp_loss.segmentation_single_pixel_binary_accuracy

        callbacks = []
        if hvd is not None:
            callbacks = callbacks + [
                # Broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),

                # Average metrics among workers at the end of every epoch.
                #
                # Note: This callback must be in the list before the ReduceLROnPlateau,
                # TensorBoard or other metrics-based callbacks.
                hvd.callbacks.MetricAverageCallback(),

                # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
                # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
                # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
            ]

        scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stopper = EarlyStopping(monitor=metric_name, min_delta=0.001, patience=32)
        csv_logger = CSVLogger(weights_name + '.csv')
        # TODO(ahundt) Re-enable results included in filename, need to fix the following error but this was skipped due to time constraints to start training:
        """
            Epoch 1/100
            6493/6494 [============================>.] - ETA: 0s - loss: 0.7651 - grasp_segmentation_single_
            pixel_accuracy: 0.5467Traceback (most recent call last):
              File "grasp_train.py", line 343, in train
                model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
              File "/home/ahundt/src/keras/keras/engine/training.py", line 1658, in fit
                validation_steps=validation_steps)
              File "/home/ahundt/src/keras/keras/engine/training.py", line 1215, in _fit_loop
                callbacks.on_epoch_end(epoch, epoch_logs)
              File "/home/ahundt/src/keras/keras/callbacks.py", line 76, in on_epoch_end
                callback.on_epoch_end(epoch, logs)
              File "/home/ahundt/src/keras/keras/callbacks.py", line 401, in on_epoch_end
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
            KeyError: 'grasp_segmentation_single_pixel_loss'
            Traceback (most recent call last):
              File "grasp_train.py", line 539, in <module>
                main()
              File "grasp_train.py", line 526, in main
                model_name=FLAGS.grasp_model)
              File "grasp_train.py", line 351, in train
                raise e
            KeyError: 'grasp_segmentation_single_pixel_loss'
        """
        # checkpoint = keras.callbacks.ModelCheckpoint(weights_name + '-epoch-{epoch:03d}-loss-{' + loss_name + ':.3f}-acc-{' + metric_name + ':.3f}.h5',
        #                                              save_best_only=True, verbose=1, monitor=metric_name)
        checkpoint = keras.callbacks.ModelCheckpoint(weights_name + '-epoch-{epoch:03d}.h5',
                                                     save_best_only=False, verbose=1)

        callbacks = callbacks + [early_stopper, csv_logger, checkpoint]

        if FLAGS.progress_tracker is 'tensorboard':
            progress_tracker = TensorBoard(log_dir='./' + weights_name, write_graph=True,
                                           write_grads=True, write_images=True)
            callbacks = callbacks + [progress_tracker]

        # progress_bar = TQDMCallback()
        # callbacks = callbacks + [progress_bar]

        # Will need to try more things later.
        # Nadam parameter choice reference:
        # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355

        # 2017-08-28 afternoon trying NADAM with higher learning rate
        # optimizer = keras.optimizers.Nadam(lr=0.03, beta_1=0.825, beta_2=0.99685)
        print('FLAGS.optimizer', FLAGS.optimizer)

        # 2017-08-28 trying SGD
        # 2017-12-18 SGD worked very well and has been the primary training optimizer from 2017-09 to 2018-01
        if FLAGS.optimizer == 'SGD':

            if hvd is not None:
                # Adjust learning rate based on number of GPUs.
                multiplier = hvd.size()
            else:
                multiplier = 1.0

            optimizer = keras.optimizers.SGD(learning_rate * multiplier)

            callbacks = callbacks + [
                # Reduce the learning rate if training plateaus.
                # TODO(ahundt) add validation checks and update monitor parameter to use them
                keras.callbacks.ReduceLROnPlateau(patience=8, verbose=1, monitor=loss_name)
            ]

        # 2017-08-27 Tried NADAM for a while with the settings below, only improved for first 2 epochs.
        # optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

        # 2017-12-18, 2018-01-04 Tried ADAM with AMSGrad, great progress initially, but stopped making progress very quickly
        if FLAGS.optimizer == 'Adam':
            optimizer = keras.optimizers.Adam(amsgrad=True)

        if hvd is not None:
            # Add Horovod Distributed Optimizer.
            optimizer = hvd.DistributedOptimizer(optimizer)

        # create the model
        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch,
            input_image_shape=input_image_shape,
            dropout_rate=dropout_rate)

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                print('Could not load weights {}, '
                      'the file does not exist, '
                      'starting fresh....'.format(load_weights))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric],
                      target_tensors=[grasp_success_op_batch])

        print('Available metrics: ' + str(model.metrics_names))

        model.summary()

        try:
            model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
            final_weights_name = weights_name + '-final.h5'
            model.save_weights(final_weights_name)
        except (Exception, KeyboardInterrupt) as e:
            # always try to save weights
            traceback.print_exc()
            final_weights_name = weights_name + '-autosaved-on-exception.h5'
            model.save_weights(final_weights_name)
            raise e
        return final_weights_name

    def eval(self, dataset=FLAGS.grasp_dataset_eval,
             batch_size=FLAGS.eval_batch_size,
             load_weights=FLAGS.load_weights,
             save_weights=FLAGS.save_weights,
             make_model_fn=grasp_model.grasp_model,
             imagenet_preprocessing=FLAGS.imagenet_preprocessing,
             grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
             grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
             resize=FLAGS.resize,
             resize_height=FLAGS.resize_height,
             resize_width=FLAGS.resize_width,
             eval_results_file=FLAGS.eval_results_file,
             model_name=FLAGS.grasp_model,
             loss=FLAGS.loss,
             metric=FLAGS.metric):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_step: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.

        # Returns

           weights_name_str or None if a new weights file was not saved.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # TODO(ahundt) ensure eval call to get_training_tensors() always runs in the same order and does not rotate the dataset.
        # list of dictionaries the length of batch_size
        (pregrasp_op_batch, grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         grasp_success_op_batch,
         num_samples) = data.get_training_tensors(batch_size=batch_size,
                                                  imagenet_preprocessing=imagenet_preprocessing,
                                                  random_crop=False,
                                                  image_augmentation=False,
                                                  resize=resize,
                                                  grasp_sequence_min_time_step=grasp_sequence_min_time_step,
                                                  grasp_sequence_max_time_step=grasp_sequence_max_time_step)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training
        csv_logger = CSVLogger(load_weights + '_eval.csv')

        # create the model
        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch,
            input_image_shape=input_image_shape,
            dropout_rate=0.0)

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                raise ValueError('Could not load weights {}, '
                                 'the file does not exist.'.format(load_weights))

        loss_name = 'loss'
        if 'segmentation_single_pixel_binary_crossentropy' in loss:
            loss = grasp_loss.segmentation_single_pixel_binary_crossentropy
            loss_name = 'segmentation_single_pixel_binary_crossentropy'

        metric_name = 'acc'
        if 'segmentation_single_pixel_binary_accuracy' in metric:
            metric_name = metric
            metric = grasp_loss.segmentation_single_pixel_binary_accuracy

        model.compile(optimizer='sgd',
                      loss=loss,
                      metrics=[metric],
                      target_tensors=[grasp_success_op_batch])

        model.summary()

        steps = float(num_samples) / float(batch_size)

        if not steps.is_integer():
            raise ValueError('The number of samples was not exactly divisible by the batch size!'
                             'For correct, reproducible evaluation your number of samples must be exactly'
                             'divisible by the batch size. Right now the batch size cannot be changed for'
                             'the last sample, so in a worst case choose a batch size of 1. Not ideal,'
                             'but manageable. num_samples: {} batch_size: {}'.format(num_samples, batch_size))

        try:
            loss, acc = model.evaluate(None, None, steps=int(steps))
            results_str = '\nevaluation results loss: ' + str(loss) + ' accuracy: ' + str(acc) + ' dataset: ' + dataset
            print(results_str)
            weights_name_str = load_weights + '_evaluation_dataset_{}_loss_{:.3f}_acc_{:.3f}'.format(dataset, loss, acc)
            weights_name_str = weights_name_str.replace('.h5', '') + '.h5'
            with open(eval_results_file, 'w') as results_file:
                results_file.write(results_str + '\n')
            if save_weights:
                model.save_weights(weights_name_str)
                print('\n saved weights with evaluation result to ' + weights_name_str)

        except KeyboardInterrupt as e:
            print('Evaluation canceled at user request, '
                  'any results are incomplete for this run.')
            return None

        return weights_name_str


def main():
    config = tf.ConfigProto()
    config.inter_op_parallelism_threads = 40
    config.intra_op_parallelism_threads = 40
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    with K.get_session() as sess:
        # Launch the training script for the particular model specified on the command line
        # or via the default flag value
        load_weights = FLAGS.load_weights
        if FLAGS.grasp_model == 'grasp_model_resnet':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_resnet(
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_pretrained':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_pretrained(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction_after_pretrained,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_single':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    depth=FLAGS.densenet_depth,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_segmentation':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_segmentation(
                    growth_rate=FLAGS.densenet_growth_rate,
                    reduction=FLAGS.densenet_reduction,
                    dense_blocks=FLAGS.densenet_dense_blocks,
                    *a, **kw)
        elif FLAGS.grasp_model == 'grasp_model_levine_2016':
            def make_model_fn(*a, **kw):
                return grasp_model.grasp_model_levine_2016(
                    *a, **kw)
        else:
            available_functions = globals()
            if FLAGS.grasp_model in available_functions:
                make_model_fn = available_functions[FLAGS.grasp_model]
            else:
                raise ValueError('unknown model selected: {}'.format(FLAGS.grasp_model))

        gt = GraspTrain()

        # train the model
        if 'train' in FLAGS.pipeline_stage:
            print('Training ' + FLAGS.grasp_model)
            load_weights = gt.train(make_model_fn=make_model_fn,
                                    load_weights=load_weights,
                                    model_name=FLAGS.grasp_model)
        # evaluate the model
        if 'eval' in FLAGS.pipeline_stage:
            print('Evaluating ' + FLAGS.grasp_model + ' on weights ' + load_weights)
            # evaluate using weights that were just computed, if available
            gt.eval(make_model_fn=make_model_fn,
                    load_weights=load_weights,
                    model_name=FLAGS.grasp_model)
        return None


if __name__ == '__main__':
    FLAGS._parse_flags()
    main()
    print('grasp_train.py run complete, original command: ', sys.argv)
    sys.exit()
