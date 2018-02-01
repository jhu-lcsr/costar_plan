#!/usr/local/bin/python
'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code based on:
    https://github.com/tnikolla/robot-grasp-detection

The raw data set resides in png and txt files located in the following structure:

    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt

image count: 885
labeled grasps count: 8019
positive: 5110 (64%)
negative: 2909 (36%)
object count: 244
object category count: 93

'''

import os
import sys
import errno
import traceback
import itertools
import six
import os
import glob
import numpy as np

import numpy as np
import tensorflow as tf
import re
from scipy.ndimage.filters import median_filter
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras._impl.keras.utils.data_utils import _hash_file
import keras
from keras import backend as K


flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('grasp_dataset', 'all', 'TODO(ahundt): integrate with brainrobotdata or allow subsets to be specified')
flags.DEFINE_boolean('grasp_download', False,
                     """Download the grasp_dataset to data_dir if it is not already present.""")
flags.DEFINE_boolean('plot', False, 'plot data in matplotlib as it is traversed')
flags.DEFINE_boolean('write', False, 'Actually write the tfrecord files if True, simply gather stats if False.')
flags.DEFINE_boolean('shuffle', True, 'shuffle the image order before running')
flags.DEFINE_boolean(
    'redundant', True,
    """Duplicate images for every bounding box so dataset is easier to traverse.
       Please note that this substantially affects the output file size,
       but the dataset parsing code becomes much easier to write.
    """)
flags.DEFINE_float('evaluate_fraction', 0.2, 'proportion of dataset to be used separately for evaluation')
flags.DEFINE_string('train_filename', 'cornell-grasping-dataset-train.tfrecord', 'filename used for the training dataset')
flags.DEFINE_string('evaluate_filename', 'cornell-grasping-dataset-evaluate.tfrecord', 'filename used for the evaluation dataset')
flags.DEFINE_string('stats_filename', 'cornell-grasping-dataset-stats.md', 'filename used for the dataset statistics file')


FLAGS = flags.FLAGS
FLAGS(sys.argv)


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


def is_sequence(arg):
    """Returns true if arg is a list or another Python Sequence, and false otherwise.

        source: https://stackoverflow.com/a/17148334/99379
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


class GraspDataset(object):
    """Cornell Grasping Dataset - about 5GB total size
        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.

        # Arguments

        data_dir: Path to dataset in TFRecord format
            (aka Example protobufs) and feature csv files.
             `~/.keras/datasets/grasping` by default.

        dataset: 'all' to load all the data.

        download: True to actually download the dataset, also see FLAGS.
    """
    def __init__(self, data_dir=None, dataset=None, download=None, verbose=0):
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir
        if dataset is None:
            dataset = FLAGS.grasp_dataset
        self.dataset = dataset
        if download is None:
            download = FLAGS.grasp_download
        if download:
            self.download(data_dir, dataset)
        self.verbose = verbose

    def download(self, data_dir=None, dataset='all'):
        '''Cornell Grasping Dataset - about 5GB total size

        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.
        Includes grasp_listing.txt with all files in all datasets;
        the feature csv files which specify the dataset size,
        the features (data channels), and the number of grasps;
        and the tfrecord files which actually contain all the data.

        If `grasp_listing_hashed.txt` is present, an additional
        hashing step will will be completed to verify dataset integrity.
        `grasp_listing_hashed.txt` will be generated automatically when
        downloading with `dataset='all'`.

        # Arguments

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.

        # Returns

           list of paths to the downloaded files

        '''
        dataset = self._update_dataset_param(dataset)
        if data_dir is None:
            if self.data_dir is None:
                data_dir = FLAGS.data_dir
            else:
                data_dir = self.data_dir
        mkdir_p(data_dir)
        print('Downloading datasets to: ', data_dir)

        url_prefix = ''
        # If a hashed version of the listing is available,
        # download the dataset and verify hashes to prevent data corruption.
        listing_hash = os.path.join(data_dir, 'grasp_listing_hash.txt')
        if os.path.isfile(listing_hash):
            files_and_hashes = np.genfromtxt(listing_hash, dtype='str', delimiter=' ')
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, file_hash=hash_str, extract=True)
                     for fpath, hash_str in tqdm(files_and_hashes)
                     if '_' + str(dataset) in fpath]
        else:
            # If a hashed version of the listing is not available,
            # simply download the dataset normally.
            listing_url = 'https://raw.githubusercontent.com/ahundt/robot-grasp-detection/master/grasp_listing.txt'
            grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=data_dir)
            grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, extract=True)
                     for fpath in tqdm(grasp_files)
                     if '_' + dataset in fpath]

            # If all files are downloaded, generate a hashed listing.
            if dataset is 'all' or dataset is '':
                print('Hashing all dataset files to prevent corruption...')
                hashes = []
                for i, f in enumerate(tqdm(files)):
                    hashes.append(_hash_file(f))
                file_hash_np = np.column_stack([grasp_files, hashes])
                with open(listing_hash, 'wb') as hash_file:
                    np.savetxt(hash_file, file_hash_np, fmt='%s', delimiter=' ', header='file_path sha256')
                print('Hashing complete, {} contains each url plus hash, and will be used to verify the '
                      'dataset during future calls to download().'.format(listing_hash))

        return files

    def _update_dataset_param(self, dataset):
        """Internal function to configure which subset of the datasets is being used.
        Helps to choose a reasonable default action based on previous user parameters.
        """
        if dataset is None and self.dataset is None:
            return []
        if dataset is 'all':
            dataset = ''
        if dataset is None and self.dataset is not None:
            dataset = self.dataset
        return dataset


def read_label_file(path):
    """
     based on https://github.com/falcondai/robot-grasp
    """
    with open(path) as f:
        xys = []
        has_nan = False
        for l in f:
            x, y = map(float, l.split())
            # some bounding boxes have invalid NaN coordinates, skip them
            if np.isnan(x) or np.isnan(y):
                has_nan = True
            xys.append((x, y))
            if len(xys) % 4 == 0 and len(xys) / 4 >= 1:
                if not has_nan:
                    yield xys[-4], xys[-3], xys[-2], xys[-1]
                has_nan = False


def bbox_info(box):
    # coordinates order y0, x0, y1, x1, ...
    box_coordinates = []

    for i in range(4):
        for j in range(2):
            box_coordinates.append(box[i][j])
    center_x = (box_coordinates[1] + box_coordinates[5])/2
    center_y = (box_coordinates[0] + box_coordinates[4])/2
    center = (center_y, center_x)
    if (box_coordinates[3] - box_coordinates[1]) == 0:
        tan = np.pi/2
    else:
        tan = (box_coordinates[2] - box_coordinates[0]) / (box_coordinates[3] - box_coordinates[1])
    angle = np.arctan2((box_coordinates[2] - box_coordinates[0]),
                       (box_coordinates[3] - box_coordinates[1]))
    width = abs(box_coordinates[5] - box_coordinates[1])
    height = abs(box_coordinates[4] - box_coordinates[0])

    return box_coordinates, center, tan, angle, width, height


def get_bbox_info_list(path_pos, path_neg):
    # list of list [y0_list, x0_list, y1_list, x1_list, ...]
    coordinates_list = [[]] * 8
    # list of centers
    center_x_list = []
    center_y_list = []
    # list of angles
    tan_list = []
    angle_list = []
    cos_list = []
    sin_list = []
    # list of width and height
    width_list = []
    height_list = []
    # list of label success/fail, 1/0
    grasp_success = []
    count_fail_success = [0, 0]

    for path_label, path in enumerate([path_neg, path_pos]):
        for box in read_label_file(path):
            coordinates, center, tan, angle, width, height = bbox_info(box)
            for i in range(8):
                coordinates_list[i].append(coordinates[i])
            center_x_list.append(center[1])
            center_y_list.append(center[0])
            tan_list.append(tan)
            angle_list.append(angle)
            cos_list.append(np.cos(angle))
            sin_list.append(np.sin(angle))
            width_list.append(width)
            height_list.append(height)
            grasp_success.append(path_label)
            count_fail_success[path_label] += 1

    return (coordinates_list, center_x_list, center_y_list, tan_list,
            angle_list, cos_list, sin_list, width_list, height_list,
            grasp_success, count_fail_success)


def gaussian_kernel_2D(size=(3, 3), center=None, sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    All coordinates are in (y, x) order, which is (height, width),
    with (0, 0) at the top left corner.

    Output with the default parameters `size=(3, 3) center=None, sigma=1`:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    Output with parameters `size=(3, 3) center=(0, 1), sigma=1`:

        [[0.60653067 1.         0.60653067]
        [0.36787945 0.60653067 0.36787945]
        [0.082085   0.13533528 0.082085  ]]

    # Arguments

        size: dimensions of the output gaussian (height_y, width_x)
        center: coordinate of the center (maximum value) of the output gaussian, (height_y, width_x).
            Default of None will automatically be the center coordinate of the output size.
        sigma: standard deviation of the gaussian in pixels

    # References:

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    # How to normalize

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    if center is None:
        center = np.array(size) / 2
    yy, xx = np.meshgrid(np.arange(size[0]),
                         np.arange(size[1]),
                         indexing='ij')
    kernel = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2. * sigma ** 2))
    return kernel


class ImageCoder(object):
    # probably based on https://github.com/visipedia/tfrecords
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})


def _process_image(filename, coder):
    # Decode the image
    with open(filename) as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def add_one_gaussian(image, center, grasp_theta, grasp_width, grasp_height, label, sigma_divisor=10):
    """ Compare to ground_truth_image in grasp_img_proc.py
    """
    sigma = max(grasp_width, grasp_height) / sigma_divisor
    # make sure center value for gaussian is 0.5
    gaussian = gaussian_kernel_2D((image.shape[0], image.shape[1]), center=center, sigma=sigma)
    # label 0 is grasp failure, label 1 is grasp success, label 0.5 will have no effect.
    # gaussian center with label 0 should be subtracting 0.5
    # gaussian center with label 1 should be adding 0.5
    gaussian = ((label * 2) - 1.0) * gaussian
    image = image + gaussian
    return image


def ground_truth_images(
        image_shape,
        grasp_cys, grasp_cxs,
        grasp_thetas,
        grasp_heights, grasp_widths,
        labels):
    gt_images = []
    if not isinstance(grasp_cys, list):
        grasp_cys = [grasp_cys]
        grasp_cxs = [grasp_cxs]
        grasp_thetas = [grasp_thetas]
        grasp_heights = [grasp_heights]
        grasp_widths = [grasp_widths]
        labels = [labels]

    for (grasp_cy, grasp_cx, grasp_theta,
         grasp_height, grasp_width, label) in zip(grasp_cys, grasp_cxs,
                                                  grasp_thetas, grasp_heights,
                                                  grasp_widths, labels):
        gt_image = np.zeros(image_shape[:2])
        gt_image = add_one_gaussian(
            gt_image, (grasp_cy, grasp_cx), grasp_theta,
            grasp_height, grasp_width, label)
        max_num = max(np.max(gt_image), 1.0)
        min_num = min(np.min(gt_image), -1.0)
        gt_image = (gt_image - min_num) / (max_num - min_num)
        gt_images += [gt_image]

    return gt_images


def visualize_example(img, center_x_list, center_y_list, grasp_success, gt_images):
    width = 3
    gt_plot_height = len(center_x_list)/2
    fig, axs = plt.subplots(gt_plot_height + 1, 4, figsize=(15, 15))
    axs[0, 0].imshow(img, zorder=0)
    # axs[0, 0].arrow(np.array(center_y_list), np.array(center_x_list),
    #                 np.array(coordinates_list[0]) - np.array(coordinates_list[2]),
    #                 np.array(coordinates_list[1]) - np.array(coordinates_list[3]), c=grasp_success)
    axs[0, 0].scatter(np.array(center_y_list), np.array(center_x_list), zorder=2, c=grasp_success, alpha=0.5, lw=2)
    axs[0, 1].imshow(img, zorder=0)
    # axs[1, 0].scatter(data[0], data[1])
    # axs[2, 0].imshow(gt_image)
    for i, gt_image in enumerate(gt_images):
        h = i % gt_plot_height + 1
        w = int(i / gt_plot_height)
        axs[h, w].imshow(img, zorder=0)
        axs[h, w].imshow(gt_image, alpha=0.75, zorder=1)
        # axs[h, w*2+1].imshow(gt_image, alpha=0.75, zorder=1)

    # axs[1, 1].hist2d(data[0], data[1])
    plt.draw()
    plt.pause(0.25)

    plt.show()
    return width


def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    """Wrapper for inserting float features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.

    https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""

    if isinstance(text, str):
        return text
    elif isinstance(text, unicode):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)


def _create_examples(filename, image_buffer, height, width, coordinates_list,
                     center_y_list, center_x_list, tan_list, angle_list,
                     sin_list, cos_list, width_list, height_list, grasp_success):
    """
    coordinates_list: a list containing 8 total lists of floats.
        Each list contains specific coordinates for the grasping box
        at that index.
        [x0, y0, x1, y1, x2, y2, x3, y3]

    This makes lists of coordinates so that images are never repeated.

    # Returns

      A list of examples
    """

    # Build an Example proto for an example
    feature = {'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(image_buffer),
               'image/height': _int64_feature(height),
               'image/width': _int64_feature(width)}
    for i in range(4):
        feature['bbox/y' + str(i)] = _floats_feature(coordinates_list[2*i])
        feature['bbox/x' + str(i)] = _floats_feature(coordinates_list[2*i+1])
    feature['bbox/cy'] = _floats_feature(center_y_list)
    feature['bbox/cx'] = _floats_feature(center_x_list)
    feature['bbox/tan'] = _floats_feature(tan_list)
    feature['bbox/theta'] = _floats_feature(angle_list)
    feature['bbox/sin_theta'] = _floats_feature(sin_list)
    feature['bbox/cos_theta'] = _floats_feature(cos_list)
    feature['bbox/width'] = _floats_feature(width_list)
    feature['bbox/height'] = _floats_feature(height_list)
    feature['bbox/grasp_success'] = _int64_feature(grasp_success)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return [example]


def _create_examples_redundant(
        filename, image_buffer, height, width, coordinates_list,
        center_y_list, center_x_list, tan_list, angle_list,
        sin_list, cos_list, width_list, height_list, grasp_success):
    """
    coordinates_list: a list containing 8 total lists of floats.
        Each list contains specific coordinates for the grasping box
        at that index.
        [x0, y0, x1, y1, x2, y2, x3, y3]

    All lists of coordinates are size 1, makes dataset easier to read
    and is not much larger because of tfrecord (protobuf)
    size optimizations.
    """
    examples = []
    for i in range(len(center_x_list)):
        # Build an Example proto for an example
        feature = {'image/filename': _bytes_feature(filename),
                   'image/encoded': _bytes_feature(image_buffer),
                   'image/height': _int64_feature(height),
                   'image/width': _int64_feature(width)}
        for j in range(4):
            feature['bbox/y' + str(j)] = _floats_feature(coordinates_list[2*j][i])
            feature['bbox/x' + str(j)] = _floats_feature(coordinates_list[2*j+1][i])
        feature['bbox/cy'] = _floats_feature(center_y_list[i])
        feature['bbox/cx'] = _floats_feature(center_x_list[i])
        feature['bbox/tan'] = _floats_feature(tan_list[i])
        feature['bbox/theta'] = _floats_feature(angle_list[i])
        feature['bbox/sin_theta'] = _floats_feature(sin_list[i])
        feature['bbox/cos_theta'] = _floats_feature(cos_list[i])
        feature['bbox/width'] = _floats_feature(width_list[i])
        feature['bbox/height'] = _floats_feature(height_list[i])
        feature['bbox/grasp_success'] = _int64_feature(grasp_success[i])
        examples += [tf.train.Example(features=tf.train.Features(feature=feature))]

    return examples


def _convert_to_examples(filename, path_pos, path_neg, image_buffer, height, width):
    """
    """
    # get the bounding box information as lists of separate floats
    (coordinates_list, center_x_list, center_y_list, tan_list,
     angle_list, cos_list, sin_list, width_list, height_list,
     grasp_success, count_fail_success) = get_bbox_info_list(path_pos, path_neg)

    attempt_count = len(center_x_list)

    if FLAGS.plot:
        gt_images = ground_truth_images([height, width], center_x_list, center_y_list, angle_list, height_list, width_list, grasp_success)
        # load the image with matplotlib for display
        img = mpimg.imread(filename)
        visualize_example(img, center_x_list, center_y_list, grasp_success, gt_images)

    create_fn = _create_examples
    if FLAGS.redundant:
        create_fn = _create_examples_redundant

    examples = create_fn(filename, image_buffer, height, width, coordinates_list, center_y_list,
                         center_x_list, tan_list, angle_list, sin_list, cos_list, width_list,
                         height_list, grasp_success)

    return examples, attempt_count, count_fail_success


def traverse_dataset(filenames, eval_fraction=FLAGS.evaluate_fraction, write=FLAGS.write, train_file=None, validation_file=None):
    coder = ImageCoder()
    image_count = len(filenames)
    train_image_count = 0
    eval_image_count = 0
    train_attempt_count = 0
    eval_attempt_count = 0
    total_attempt_count = 0
    train_fail_success_count = [0, 0]
    eval_fail_success_count = [0, 0]
    steps_per_eval = int(np.ceil(1.0 / eval_fraction))

    if write:
        writer_train = tf.python_io.TFRecordWriter(train_file)
        writer_validation = tf.python_io.TFRecordWriter(validation_file)

    for i, filename in enumerate(tqdm(filenames)):
        bbox_pos_path = filename[:-5]+'cpos.txt'
        bbox_neg_path = filename[:-5]+'cneg.txt'
        image_buffer, height, width = _process_image(filename, coder)
        examples, attempt_count, count_fail_success = _convert_to_examples(
            filename, bbox_pos_path, bbox_neg_path, image_buffer, height, width)

        # Split the dataset in 80% for training and 20% for validation
        total_attempt_count += attempt_count
        if i % steps_per_eval == 0:
            eval_image_count += 1
            eval_attempt_count += attempt_count
            eval_fail_success_count[0] += count_fail_success[0]
            eval_fail_success_count[1] += count_fail_success[1]
            if write:
                for example in examples:
                    writer_validation.write(example.SerializeToString())
        else:
            train_image_count += 1
            train_attempt_count += attempt_count
            train_fail_success_count[0] += count_fail_success[0]
            train_fail_success_count[1] += count_fail_success[1]
            if write:
                for example in examples:
                    writer_train.write(example.SerializeToString())

    if write:
        writer_train.close()
        writer_validation.close()

    return (image_count, total_attempt_count, train_image_count, eval_image_count,
            train_attempt_count, eval_attempt_count, train_fail_success_count,
            eval_fail_success_count)


def get_cornell_grasping_dataset_filenames(data_dir=FLAGS.data_dir, shuffle=FLAGS.shuffle):
    # Creating a list with all the image paths
    folders = range(1, 11)
    folders = ['0'+str(i) if i < 10 else '10' for i in folders]
    png_filenames = []

    for i in folders:
        for name in glob.glob(os.path.join(data_dir, i, 'pcd' + i + '*r.png')):
            png_filenames.append(name)

    if shuffle:
        # Shuffle the list of image paths
        np.random.shuffle(png_filenames)

    bbox_successful_filenames = []
    bbox_failure_filenames = []
    for filename in png_filenames:
        bbox_pos_path = filename[:-5]+'cpos.txt'
        bbox_neg_path = filename[:-5]+'cneg.txt'
        bbox_successful_filenames += [bbox_pos_path]
        bbox_successful_filenames += [bbox_neg_path]
    return png_filenames, bbox_successful_filenames, bbox_failure_filenames


def get_stat(name, amount, total, percent_description=''):
    return ' - %s %s, ' % (amount, name) + "{0:.2f}".format(100.0 * amount/total) + ' percent' + percent_description


def main():

    # plt.ion()
    gd = GraspDataset()
    if FLAGS.grasp_download:
        gd.download(dataset=FLAGS.grasp_dataset)

    # Creating a list with all the image paths
    png_filenames, _, _ = get_cornell_grasping_dataset_filenames()

    train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
    validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)
    stats_file = os.path.join(FLAGS.data_dir, FLAGS.stats_filename)
    print(train_file)
    print(validation_file)
    if not FLAGS.write:
        print('WARNING: Gathering stats that WILL NOT BE WRITTEN TO A FILE'
              ' training and evaluation stats will not '
              'be valid for any existing tfrecord file.'
              'To write to a file run python build_cgd_dataset.py --write.')

    (image_count, total_attempt_count, train_image_count, eval_image_count,
     train_attempt_count, eval_attempt_count, train_fail_success_count,
     eval_fail_success_count) = traverse_dataset(png_filenames, train_file=train_file, validation_file=validation_file)

    total_success_count = train_fail_success_count[1] + eval_fail_success_count[1]
    total_fail_count = train_fail_success_count[0] + eval_fail_success_count[0]

    stat_string = ''

    stat_string += '\n' + ('Cornell Grasping Dataset')
    stat_string += '\n' + ('------------------------')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('TFRecord generation complete. Saved files:\n\n - %s\n - %s\n - %s' % (train_file, validation_file, stats_file))
    stat_string += '\n' + ('')
    stat_string += '\n' + ('Dataset Statistics')
    stat_string += '\n' + ('---------------')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Totals')
    stat_string += '\n' + (' - %s images' % image_count)
    stat_string += '\n' + (' - %s grasp attempts' % total_attempt_count)
    stat_string += '\n' + get_stat('successful grasps', total_success_count, total_attempt_count)
    stat_string += '\n' + get_stat('failed grasps', total_fail_count, total_attempt_count)
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Training Data')
    stat_string += '\n' + get_stat('images', train_image_count, image_count)
    stat_string += '\n' + get_stat('grasp attempts', train_attempt_count, total_attempt_count, ' of total')
    stat_string += '\n' + get_stat('successful grasps', train_fail_success_count[1], train_attempt_count, ' of training data')
    stat_string += '\n' + get_stat('failed grasps', train_fail_success_count[0], train_attempt_count, ' of training data')
    stat_string += '\n' + ('')
    stat_string += '\n' + ('### Evaluation Data')
    stat_string += '\n' + get_stat('images', eval_image_count, image_count)
    stat_string += '\n' + get_stat('grasp attempts', eval_attempt_count, total_attempt_count, ' of toal')
    stat_string += '\n' + get_stat('successful grasps', eval_fail_success_count[1], eval_attempt_count, ' of eval data')
    stat_string += '\n' + get_stat('failed grasps', eval_fail_success_count[0], eval_attempt_count, ' of eval data')
    stat_string += '\n' + ('')

    if FLAGS.write:
        with open(FLAGS.stats_filename, "w") as text_file:
            text_file.write(stat_string)

    print(stat_string)

if __name__ == '__main__':
    main()
