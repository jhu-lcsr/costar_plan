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
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from grasp_model import concat_images_with_tiled_vector_layer
from grasp_model import top_block
from grasp_model import create_tree_roots
from grasp_model import dilated_late_concat_model
from cornell_grasp_dataset_reader import parse_and_preprocess
# https://github.com/aurora95/Keras-FCN
# TODO(ahundt) move keras_fcn directly into this repository, into keras-contrib, or make a proper installer
from keras_contrib.applications.fully_convolutional_networks import AtrousFCN_Vgg16_16s

import grasp_loss as grasp_loss


flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('grasp_dataset', 'all', 'TODO(ahundt): integrate with brainrobotdata or allow subsets to be specified')
flags.DEFINE_boolean('grasp_download', False,
                     """Download the grasp_dataset to data_dir if it is not already present.""")
flags.DEFINE_string('train_filename', 'cornell-grasping-dataset-train.tfrecord', 'filename used for the training dataset')
flags.DEFINE_string('evaluate_filename', 'cornell-grasping-dataset-evaluate.tfrecord', 'filename used for the evaluation dataset')


flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Initial learning rate.'
)
flags.DEFINE_integer(
    'num_epochs',
    300,
    'Number of epochs to run trainer.'
)
flags.DEFINE_integer(
    'batch_size',
    64,
    'Batch size.'
)
flags.DEFINE_string(
    'log_dir',
    '/tmp/tf',
    'Tensorboard log_dir.'
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


def dilated_vgg_model(
        images=None, vectors=None,
        image_shapes=None, vector_shapes=None,
        dropout_rate=None,
        vector_dense_filters=256,
        dilation_rate=2,
        activation='sigmoid',
        final_pooling=None,
        include_top=True,
        top='classification',
        classes=1,
        output_shape=None,
        trainable=False,
        verbose=0,
        image_model_name='vgg'):
    with K.name_scope('dilated_' + image_model_name) as scope:
        # VGG16 weights are shared and not trainable
        if top == 'segmentation':
            vgg_model = AtrousFCN_Vgg16_16s(
                input_shape=image_shapes[0], include_top=False,
                classes=classes, upsample=False)
        else:
            if image_model_name == 'vgg':
                vgg_model = keras.applications.vgg16.VGG16(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
            elif image_model_name == 'nasnet':
                vgg_model = keras.applications.nasnet.NASNetLarge(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes
                )
            elif image_model_name == 'resnet':
                vgg_model = keras.applications.resnet50.ResNet50(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
            elif image_model_name == 'densenet':
                vgg_model = keras.applications.densenet.DenseNet169(
                    input_shape=image_shapes[0], include_top=False,
                    classes=classes)
            else:
                raise ValueError('Unsupported image_model_name')

        if not trainable:
            for layer in vgg_model.layers:
                layer.trainable = False

        def create_vgg_model(tensor):
            """ Image classifier weights are shared.
            """
            return vgg_model(tensor)

        def vector_branch_dense(tensor, vector_dense_filters=vector_dense_filters):
            """ Vector branches that simply contain a single dense layer.
            """
            return Dense(vector_dense_filters)(tensor)

        model = dilated_late_concat_model(
            images=images, vectors=vectors,
            image_shapes=image_shapes, vector_shapes=vector_shapes,
            dropout_rate=dropout_rate,
            vector_dense_filters=vector_dense_filters,
            create_image_tree_roots_fn=create_vgg_model,
            create_vector_tree_roots_fn=vector_branch_dense,
            dilation_rate=dilation_rate,
            activation=activation,
            final_pooling=final_pooling,
            include_top=include_top,
            top=top,
            classes=classes,
            output_shape=output_shape,
            verbose=verbose
        )
    return model


class PrintLogsCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('\nlogs:', logs)


def run_training(learning_rate=0.001, batch_size=10, num_gpus=1, top='classification', epochs=300, preprocessing_mode=None):
    """

    top: options are 'segmentation' and 'classification'.
    """

    data_features = ['image/preprocessed', 'bbox/preprocessed/cy_cx_normalized_2',
                     'bbox/preprocessed/sin_cos_2']
    # see parse_and_preprocess() for the creation of these features
    if top == 'segmentation':
        label_features = ['grasp_success_yx_3']
        monitor_loss_name = 'segmentation_gaussian_binary_crossentropy'
        monitor_metric_name = 'val_segmentation_single_pixel_binary_accuracy'
        loss = grasp_loss.segmentation_gaussian_binary_crossentropy
        metrics = [grasp_loss.segmentation_single_pixel_binary_accuracy, grasp_loss.mean_pred]
        model_name = 'dilated_vgg_model'
        image_model_name = 'vgg'
    else:
        label_features = ['grasp_success']
        monitor_metric_name = 'val_binary_accuracy'
        monitor_loss_name = 'val_loss'
        metrics = ['binary_accuracy', grasp_loss.mean_pred, grasp_loss.mean_true]
        loss = keras.losses.binary_crossentropy
        model_name = 'vgg_model'
        image_model_name = 'vgg'

    # If loading pretrained weights
    # it is very important to preprocess
    # in exactly the same way the model
    # was originally trained
    if preprocessing_mode is None:
        if 'densenet' in model_name:
            preprocessing_mode = 'torch'
        else:
            preprocessing_mode = 'caffe'


    # create dilated_vgg_model with inputs [image], [sin_theta, cos_theta]
    # TODO(ahundt) split vector shapes up appropriately for dense layers in dilated_late_concat_model
    model = dilated_vgg_model(
        image_shapes=[(FLAGS.sensor_image_height, FLAGS.sensor_image_width, 3)],
        vector_shapes=[(4,)],
        dropout_rate=0.5,
        top=top,
        image_model_name=image_model_name)

    print(monitor_loss_name)
    # TODO(ahundt) add a loss that changes size with how open the gripper is
    # loss = grasp_loss.segmentation_gaussian_measurement

    save_weights = ''
    dataset_names_str = 'cornell_grasping'
    weights_name = timeStamped(save_weights + '-' + model_name + '-dataset_' + dataset_names_str + '-' + label_features[0])
    callbacks = []

    optimizer = keras.optimizers.SGD(learning_rate * 1.0)
    callbacks = callbacks + [
        # Reduce the learning rate if training plateaus.
        keras.callbacks.ReduceLROnPlateau(patience=20, verbose=1, factor=0.5, monitor=monitor_loss_name)
    ]

    csv_logger = CSVLogger(weights_name + '.csv')
    callbacks = callbacks + [csv_logger]
    callbacks += [PrintLogsCallback()]
    log_dir = './logs_cornell/' + weights_name
    print('Enabling tensorboard in ' + log_dir)
    mkdir_p(log_dir)

    checkpoint = keras.callbacks.ModelCheckpoint(log_dir + weights_name + '-epoch-{epoch:03d}-' +
                                                 monitor_loss_name + '-{' + monitor_loss_name + ':.3f}-' +
                                                 monitor_metric_name + '-{' + monitor_metric_name + ':.3f}.h5',
                                                 save_best_only=True, verbose=1, monitor=monitor_metric_name)
    callbacks = callbacks + [checkpoint]
    # An additional useful param is write_batch_performance:
    #  https://github.com/keras-team/keras/pull/7617
    #  write_batch_performance=True)
    progress_tracker = TensorBoard(log_dir=log_dir, write_graph=True,
                                   write_grads=True, write_images=True,
                                   histogram_freq=1, batch_size=batch_size)
                                   # histogram_freq=0, batch_size=batch_size,
                                   # write_batch_performance=True)
    callbacks = callbacks + [progress_tracker]
    train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
    validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)
    # TODO(ahundt) WARNING: THE NUMBER OF TRAIN/VAL STEPS VARIES EVERY TIME THE DATASET IS CONVERTED, AUTOMATE SETTING THOSE NUMBERS
    samples_in_training_dataset = 6402
    samples_in_val_dataset = 1617
    val_batch_size = 11
    steps_in_val_dataset, divides_evenly = np.divmod(samples_in_val_dataset, val_batch_size)
    assert divides_evenly == 0
    steps_per_epoch_train = np.ceil(float(samples_in_training_dataset) / float(batch_size))

    if num_gpus > 1:
        parallel_model = keras.utils.multi_gpu_model(model, num_gpus)
    else:
        parallel_model = model

    parallel_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    # i = 0
    # for batch in tqdm(
    #     cornell_grasp_dataset_reader.yield_record(
    #         train_file, label_features, data_features,
    #         batch_size=batch_size,
    #         parse_example_proto_fn=choose_parse_example_proto_fn()))):
    #     i += 1

    # Get the validation dataset in one big numpy array for validation
    # This lets us take advantage of tensorboard visualization
    all_validation_data = next(cornell_grasp_dataset_reader.yield_record(
        validation_file, label_features, data_features,
        is_training=False, batch_size=samples_in_val_dataset,
        parse_example_proto_fn=parse_and_preprocess))

    parallel_model.fit_generator(
        cornell_grasp_dataset_reader.yield_record(
            train_file, label_features, data_features,
            batch_size=batch_size,
            parse_example_proto_fn=parse_and_preprocess),
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=all_validation_data,
        validation_steps=steps_in_val_dataset,
        callbacks=callbacks)


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
                  [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)
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
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
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
