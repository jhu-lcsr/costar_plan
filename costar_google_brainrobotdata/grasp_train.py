"""Code for building the input for the prediction model."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

import moviepy.editor as mpy

FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3
BATCH_SIZE = 25


def build_image_input(sess, train=True, novel=True):
    """Create input tfrecord tensors.

    Args:
      novel: whether or not to grab novel or seen images.
    Returns:
      list of tensors corresponding to images. The images
      tensor is 5D, batch x time x height x width x channels.
    Raises:
      RuntimeError: if no files found.
    """
    if train:
        data_dir = os.path.expanduser('~/Downloads/google_brainrobotdata_grasp')
    elif novel:
        data_dir = os.path.expanduser('~/Downloads/google_brainrobotdata_grasp')
    else:
        data_dir = os.path.expanduser('~/Downloads/google_brainrobotdata_grasp')

    feature_csv_files = gfile.Glob(os.path.join(data_dir, '*.csv*'))
    for feature_csv_file in feature_csv_files:
        print(feature_csv_file)
        features = np.genfromtxt(feature_csv_file, dtype=str)
        feature_count = int(features[0])
        attempt_count = int(features[1])
        filenames = gfile.Glob(os.path.join(data_dir, '*{}.tfrecord*-of-*'.format(feature_count)))

        # filenames = ['/Users/athundt/Downloads/google_brainrobotdata_grasp/grasping_dataset_052.tfrecord']
        # filenames = ['/Users/athundt/Downloads/google_brainrobotdata_grasp/grasping_dataset_102.tfrecord-00000-of-00219']

        print(filenames)
        if not filenames:
            raise RuntimeError('No data files found.')
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        image_seq = []

        num_grasp_steps_name = 'num_grasp_steps'
        images_feature_names = [str(image_feature_name) for image_feature_name in features if '/image/' in image_feature_name]
        for image_name in images_feature_names:
            features_dict = {image_name: tf.FixedLenFeature([1], tf.string),
                             num_grasp_steps_name: tf.FixedLenFeature([1], tf.string)}
            features = tf.parse_single_example(serialized_example, features=features_dict)
            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
            image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

            image = tf.reshape(image, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            # image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
            image_seq.append(image)

    image_seq = tf.concat(image_seq, 0)

    image_batch = tf.train.batch(
        [image_seq],
        BATCH_SIZE,
        num_threads=1,
        capacity=1)

    return image_batch


def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

sess = tf.InteractiveSession()
train_image_tensor = build_image_input(sess)
tf.train.start_queue_runners(sess)
sess.run(tf.global_variables_initializer())
train_videos = sess.run(train_image_tensor)

for i in range(BATCH_SIZE):
    video = train_videos[i]
    npy_to_gif(video, '~/grasp_' + str(i) + '.gif')

