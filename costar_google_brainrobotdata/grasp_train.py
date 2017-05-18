"""Code for building the input for the prediction model."""

import os
import errno

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from keras.utils import get_file

import moviepy.editor as mpy

tf.flags.DEFINE_string('data_dir', os.path.join(os.path.expanduser("~"),
                       '.keras', 'datasets', 'grasping'),
                       """Path to dataset in TFRecord format
                       (aka Example protobufs) and feature csv files.""")
tf.flags.DEFINE_integer('batch_size', 25, 'batch size per compute device')
tf.flags.DEFINE_integer('sensor_image_width', 640, 'Camera Image Width')
tf.flags.DEFINE_integer('sensor_image_height', 512, 'Camera Image Height')
tf.flags.DEFINE_integer('sensor_color_channels', 3, 'Number of color channels (3, RGB)')

FLAGS = flags.FLAGS

def mkdir_p(path):
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class GraspDataset:
    """Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Downloads to `~/.keras/datasets/grasping` by default.

    """
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir

    def download(self, dataset=''):
        '''Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Downloads to `~/.keras/datasets/grasping` by default.

        # Arguments

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.

        '''
        mkdir_p(FLAGS.data_dir)
        print(FLAGS.data_dir)
        listing_url = 'https://sites.google.com/site/brainrobotdata/home/grasping-dataset/grasp_listing.txt'
        grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=self.data_dir)
        grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
        url_prefix = 'https://storage.googleapis.com/brain-robotics-data/'
        files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=self.data_dir)
                 for fpath in grasp_files
                 if dataset in fpath]
        return files

    def get_feature_csv_files(self):
        return gfile.Glob(os.path.join(os.path.expanduser(self.data_dir), '*.csv*'))

    def get_tfrecord_filenames(self, feature_csv_file):
        print(feature_csv_file)
        features = np.genfromtxt(feature_csv_file, dtype=str)
        feature_count = int(features[0])
        attempt_count = int(features[1])
        features = features[2:]
        tfrecord_paths = gfile.Glob(os.path.join(os.path.expanduser(FLAGS.data_dir), '*{}.tfrecord*-of-*'.format(feature_count)))
        return features, feature_count, attempt_count, tfrecord_paths

    def build_image_input(self, sess, train=True, novel=True):
        """Create input tfrecord tensors.

        Args:
        novel: whether or not to grab novel or seen images.
        Returns:
        list of tensors corresponding to images. The images
        tensor is 5D, batch x time x height x width x channels.
        Raises:
        RuntimeError: if no files found.
        """
        feature_csv_files = self.get_feature_csv_files()
        for feature_csv_file in feature_csv_files:
            features, feature_count, attempt_count, tfrecord_paths = self.get_tfrecord_filenames(feature_csv_file)
            print(tfrecord_paths)
            if not tfrecord_paths:
                raise RuntimeError('No data files found.')
            filename_queue = tf.train.string_input_producer(tfrecord_paths, shuffle=False)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            image_seq = []

            num_grasp_steps_name = 'num_grasp_steps'
            images_feature_names = []
            # some silly tricks to get the feature names in the right order while
            # allowing variation between the various datasets
            images_feature_names.extend([str(ifname) for ifname in features if ('grasp/image/encoded' in ifname) and not ('post' in ifname)])
            for i in range(10):
                fstr = 'grasp/{}/image/encoded'.format(i)
                print(fstr)
                images_feature_names.extend([str(ifname) for ifname in features if (fstr in ifname)])
            images_feature_names.extend([str(ifname) for ifname in features if ('post_grasp/image/encoded' in ifname)])
            images_feature_names.extend([str(ifname) for ifname in features if ('present/image/encoded' in ifname)])
            images_feature_names.extend([str(ifname) for ifname in features if ('post_drop/image/encoded' in ifname)])
            print(images_feature_names)
            for image_name in images_feature_names:
                features_dict = {image_name: tf.FixedLenFeature([1], tf.string),
                                num_grasp_steps_name: tf.FixedLenFeature([1], tf.string)}
                features = tf.parse_single_example(serialized_example, features=features_dict)
                image_buffer = tf.reshape(features[image_name], shape=[])
                image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.sensor_color_channels)
                image.set_shape([FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels])

                image = tf.reshape(image, [1, FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels])
                # image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
                image_seq.append(image)

        image_seq = tf.concat(image_seq, 0)

        image_batch = tf.train.batch(
            [image_seq],
            FLAGS.batch_size,
            num_threads=1,
            capacity=1)

        return image_batch

    def npy_to_gif(self, npy, filename):
        clip = mpy.ImageSequenceClip(list(npy), fps=2)
        clip.write_gif(filename)

    def create_gif(self, sess):
        train_image_tensor = self.build_image_input(sess)
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        train_videos = sess.run(train_image_tensor)

        for i in range(FLAGS.batch_size):
            video = train_videos[i]
            self.npy_to_gif(video, '~/grasp_' + str(i) + '.gif')


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    gd = GraspDataset()
    # gd.download()
    gd.create_gif(sess)

