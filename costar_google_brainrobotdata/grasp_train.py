import os
import datetime
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import grasp_dataset
import grasp_model

from tensorflow.python.platform import flags


tf.flags.DEFINE_string('grasp_model', 'grasp_model_single',
                       """Choose the model definition to run, options are grasp_model and grasp_model_segmentation""")
tf.flags.DEFINE_string('save_weights', 'grasp_model_weights',
                       """Save a file with the trained model weights.""")
tf.flags.DEFINE_string('load_weights', 'grasp_model_weights',
                       """Load and continue training the specified file containing model weights.""")
tf.flags.DEFINE_integer('epochs', 20,
                        """Epochs of training""")
# tf.flags.DEFINE_integer('batch_size', 1,
#                         """size of a single batch during training""")

FLAGS = flags.FLAGS


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class GraspTrain(object):

    def train(self, dataset=FLAGS.grasp_dataset, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
              load_weights=FLAGS.save_weights,
              save_weights=FLAGS.load_weights,
              make_model_fn=grasp_model.grasp_model,
              imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
              grasp_sequence_max_time_steps=FLAGS.grasp_sequence_max_time_steps,
              random_crop=FLAGS.random_crop,
              resize=FLAGS.resize,
              resize_height=FLAGS.resize_height,
              resize_width=FLAGS.resize_width):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_max_time_steps: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        (pregrasp_op_batch, grasp_step_op_batch,
         simplified_grasp_command_op_batch,
         example_batch_size,
         grasp_success_op_batch,
         num_samples) = self.single_pose_training_tensors(batch_size,
                                                          imagenet_mean_subtraction,
                                                          random_crop,
                                                          resize,
                                                          grasp_sequence_max_time_steps)

        if resize:
            input_image_shape = [resize_height, resize_width, 3]
        else:
            input_image_shape = [512, 640, 3]

        ########################################################
        # End tensor configuration, begin model configuration and training

        weights_name = timeStamped(save_weights)
        learning_rate = 0.025
        learning_power_decay_rate = 0.4

        # ###############learning rate scheduler####################
        # source: https://github.com/aurora95/Keras-FCN/blob/master/train.py
        def lr_scheduler(epoch, mode='power_decay'):
            '''if lr_dict.has_key(epoch):
                lr = lr_dict[epoch]
                print 'lr: %f' % lr'''

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

        scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stopper = EarlyStopping(monitor='acc', min_delta=0.001, patience=10)
        csv_logger = CSVLogger(weights_name + '.csv')
        checkpoint = keras.callbacks.ModelCheckpoint(weights_name + '.epoch-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}.h5',
                                                     save_best_only=True, verbose=1, monitor='acc')

        callbacks = [scheduler, early_stopper, csv_logger, checkpoint]

        # Will need to try more things later.
        # Nadam parameter choice reference:
        # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355

        # 2017-08-28 afternoon trying NADAM with higher learning rate
        # optimizer = keras.optimizers.Nadam(lr=0.03, beta_1=0.825, beta_2=0.99685)

        # 2017-08-28 trying SGD
        optimizer = keras.optimizers.SGD(lr=learning_rate)

        # 2017-08-27
        # Tried NADAM for a while with the settings below, only improved for first 2 epochs.
        # optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

        # create the model
        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch,
            input_image_shape=input_image_shape,
            batch_size=example_batch_size)

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                print('Could not load weights {}, '
                      'the file does not exist, '
                      'starting fresh....'.format(load_weights))

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[grasp_success_op_batch])

        model.summary()

        # make sure we visit every image once
        steps_per_epoch = int(np.ceil(float(num_samples)/float(batch_size)))

        try:
            model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        except KeyboardInterrupt, e:
            # save weights if the user asked to end training
            pass
        model.save_weights(weights_name + '_final.h5')


if __name__ == '__main__':

    with K.get_session() as sess:
        if FLAGS.grasp_model is 'grasp_model_single':
            model_fn = grasp_model.grasp_model
        elif FLAGS.grasp_model is 'grasp_model_segmentation':
            model_fn = grasp_model.grasp_model_segmentation
        else:
            raise ValueError('unknown model selected: {}'.format(FLAGS.grasp_model))

        gt = GraspTrain()
        gt.train(make_model_fn=model_fn)
