#!/usr/local/bin/python
'''
Image processing of cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import copy
import math
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

import tensorflow as tf
from tensorflow.python.platform import flags

import keras
from keras import backend as K
import keras_contrib

from grasp_loss import gaussian_kernel_2D
from inception_preprocessing import preprocess_image
import random_crop as rcp

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
flags.DEFINE_string('test_filename', 'cornell-grasping-dataset-test.tfrecord', 'filename used for the evaluation dataset')
flags.DEFINE_integer('image_size', 224,
                     """DEPRECATED - this doesn't do anything right now. Provide square images of this size.""")
flags.DEFINE_integer('num_preprocess_threads', 12,
                     """Number of preprocessing threads per tower. """
                     """Please make this a multiple of 4.""")
flags.DEFINE_integer('num_readers', 20,
                     """Number of parallel threads reading from the dataset.""")
flags.DEFINE_integer('input_queue_memory_factor', 12,
                     """Size of the queue of preprocessed images. """
                     """Default is ideal but try smaller values, e.g. """
                     """4, 2 or 1, if host memory is constrained. See """
                     """comments in code for more details.""")
flags.DEFINE_boolean(
    'redundant', True,
    """Duplicate images for every bounding box so dataset is easier to traverse.
       Please note that this does not substantially affect file size because
       protobuf is the underlying TFRecord data type and it
       has optimizations eliminating repeated identical data entries.
       See cornell_grasp_dataset_writer.py for more details.
    """)
flags.DEFINE_boolean('showTextBox', True,
                     """Display textBox with bbox info on image.
                     """)
flags.DEFINE_integer('sigma_divisor', 10,
                     """Sigma divisor for grasp success 2d labels.""")
flags.DEFINE_integer(
    'sensor_image_height',
    480,
    'The height of the dataset images'
)
flags.DEFINE_integer(
    'sensor_image_width',
    640,
    'The width of the dataset images'
)
flags.DEFINE_integer(
    'sensor_color_channels',
    3,
    'The width of the dataset images'
)
flags.DEFINE_integer('crop_height', 224,
                     """Height to crop images, resize_width and resize_height is applied next""")
flags.DEFINE_integer('crop_width', 224,
                     """Width to crop images, resize_width and resize_height is applied next""")
flags.DEFINE_boolean('random_crop', True,
                     """random_crop will apply the tf random crop function with
                        the parameters specified by crop_width and crop_height.

                        If random crop is disabled, a fixed crop will be applied
                        to a box on the far right of the image which is vertically
                        centered. If no crop is desired simply set the crop_width
                        and crop_height equal to the sensor_image_width and
                        sensor_image_height. However, it is important to ensure
                        that you crop images to the same size during both training
                        and test time.

                        If crop_to_gripper and random_crop are enabled at the same
                        time, the translation will be limited to the width and height
                        of the grasp rectangle bounding box.
                     """)
flags.DEFINE_boolean('crop_to_gripper', True,
                     """crop to gripper will project and crop the image around
                        the proposed gripper position and orientation.

                        crop_to_gripper and random_crop cannot be enabled at the same time.
                     """)
flags.DEFINE_boolean(
    'crop_to_gripper_and_rotate', True,
    """Should the rotation be applied to the image
       in addition to cropping to the gripper when crop_to_gripper is True?

       This can be True when both crop_to_gripper and random_crop are True.
       This makes the gripper pose always be aligned as if the plates were
       vertically oriented for a horizontal grasp of the image.

       Has no effect when crop_to_gripper is False.
    """
)
flags.DEFINE_boolean(
    'random_rotation', True,
    """ Apply a random rotation to the input data.

        TODO(ahundt) integrate this option, currently has no effect.

        If crop_to_gripper_and_rotate is False, an arbitrary rotation will be selected.
        If crop_to_gripper_and_rotate is True, this will be limited to 15 degrees in either direction.
    """
)
flags.DEFINE_integer('resize_height', 224,
                     """Height to resize images before prediction, if enabled.""")
flags.DEFINE_integer('resize_width', 224,
                     """Width to resize images before prediction, if enabled.""")
flags.DEFINE_boolean('resize', False,
                     """resize will resize the input images to the desired dimensions specified by the
                        resize_width and resize_height flags. It is suggested that an exact factor of 2 be used
                        relative to the input image directions if random_crop is disabled or the crop dimensions otherwise.
                     """)

FLAGS = flags.FLAGS


def parse_example_proto(examples_serialized, have_image_id=False):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64)
    }

    # TODO(ahundt) remove boolean once we are set up with k-fold cross validation of images and objects
    if have_image_id:
        feature_map['object/id'] = tf.FixedLenFeature([], dtype=tf.int64)

    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        feature_map[y_key] = tf.VarLenFeature(dtype=tf.float32)
        feature_map[x_key] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cy'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cx'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/tan'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/width'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/height'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.VarLenFeature(dtype=tf.int64)
    # feature_map['bbox/sin_2_theta'] = tf.sin(feature_map['bbox/theta'] * 2.0)
    # feature_map['bbox/cos_2_theta'] = tf.cos(feature_map['bbox/theta'] * 2.0)

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def parse_example_proto_redundant(examples_serialized, have_image_id=False):
    """ Parse data from the tfrecord

    See also: _create_examples_redundant()
    """
    feature_map = {
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64)
    }

    # TODO(ahundt) remove boolean once we are set up with k-fold cross validation of images and objects
    if have_image_id:
        feature_map['image/id'] = tf.FixedLenFeature([], dtype=tf.int64)

    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        feature_map[y_key] = tf.FixedLenFeature([1], dtype=tf.float32)
        feature_map[x_key] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cy'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cx'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/tan'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/width'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/height'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.FixedLenFeature([1], dtype=tf.int64)

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def eval_image(image, height, width):
    # TODO(ahundt) THE DIMENSIONS/COORDINATES AREN'T RIGHT HERE, FIX IT!
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])

    return image


def distort_color(image, thread_id):
    color_ordering = thread_id % 2
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def distort_image(image, height, width, thread_id):
    # Need to update coordinates if flipping
    # distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(image, thread_id)
    return distorted_image


def image_preprocessing(image_buffer, train, thread_id=0):
    height = FLAGS.image_size
    width = FLAGS.image_size
    image = tf.image.decode_png(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.resize_images(image, [height, width])
    if train:
        image = distort_image(image, height, width, thread_id)
    # else:
    #    image = eval_image(image, height, width)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    return image

# from math import pi, cos, sin


# class Vector:
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __add__(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return Vector(self.x + v.x, self.y + v.y)

#     def __sub__(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return Vector(self.x - v.x, self.y - v.y)

#     def cross(self, v):
#         if not isinstance(v, Vector):
#             return NotImplemented
#         return self.x*v.y - self.y*v.x


# class Line:
#     # ax + by + c = 0
#     def __init__(self, v1, v2):
#         self.a = v2.y - v1.y
#         self.b = v1.x - v2.x
#         self.c = v2.cross(v1)

#     def __call__(self, p):
#         return self.a*p.x + self.b*p.y + self.c

#     def intersection(self, other):
#         # http://www.mathopenref.com/coordpolygonarea.html
#         # https://stackoverflow.com/a/45268241/99379
#         # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
#         if not isinstance(other, Line):
#             return NotImplemented
#         w = self.a*other.b - self.b*other.a
#         return Vector(
#             (self.b*other.c - self.c*other.b)/w,
#             (self.c*other.a - self.a*other.c)/w
#         )


# def rectangle_vertices(cx, cy, w, h, r):
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     angle = pi*r/180
#     dx = w/2
#     dy = h/2
#     dxcos = dx*cos(angle)
#     dxsin = dx*sin(angle)
#     dycos = dy*cos(angle)
#     dysin = dy*sin(angle)
#     return (
#         Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
#         Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
#         Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
#         Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
#     )

# def intersection_area(r1, r2):
#     # http://www.mathopenref.com/coordpolygonarea.html
#     # https://stackoverflow.com/a/45268241/99379
#     # r1 and r2 are in (center, width, height, rotation) representation
#     # First convert these into a sequence of vertices

#     rect0 = rectangle_vertices(*r1)
#     rect1 = rectangle_vertices(*r2)

#     # Use the vertices of the first rectangle as
#     # starting vertices of the intersection polygon.
#     rect0 = rect0

#     # Loop over the edges of the second rectangle
#     for p, q in zip(rect1, rect1[1:] + rect1[:1]):
#         if len(rect0) <= 2:
#             break # No intersection

#         line = Line(p, q)

#         # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
#         # any point p with line(p) > 0 is on the "outside".

#         # Loop over the edges of the rect0 polygon,
#         # and determine which part is inside and which is outside.
#         new_intersection = []
#         line_values = [line(t) for t in rect0]
#         for s, t, s_value, t_value in zip(
#                 rect0, rect0[1:] + rect0[:1],
#                 line_values, line_values[1:] + line_values[:1]):
#             if s_value <= 0:
#                 new_intersection.append(s)
#             if s_value * t_value < 0:
#                 # Points are on opposite sides.
#                 # Add the intersection of the lines to new_intersection.
#                 intersection_point = line.intersection(Line(s, t))
#                 new_intersection.append(intersection_point)

#         intersection = new_intersection

#     # Calculate area
#     if len(intersection) <= 2:
#         return 0

#     return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
#                      zip(intersection, intersection[1:] + intersection[:1]))


# intersection_area(r0y0, r0x0, r0y1, r0x1, r0y2, r0x2, r0y3, r0x3, r1y0, r1x0, r1y1, r1x1, r1y2, r1x2,  r1y3, r1x3):
def rectangle_points(r0y0, r0x0, r0y1, r0x1, r0y2, r0x2, r0y3, r0x3):
    p0yx = K.concatenate([r0y0, r0x0])
    p1yx = K.concatenate([r0y1, r0x1])
    p2yx = K.concatenate([r0y2, r0x2])
    p3yx = K.concatenate([r0y3, r0x3])
    return [p0yx, p1yx, p2yx, p3yx]


def rectangle_vectors(rp):
    """
    # Arguments

    rp: rectangle points [p0yx, p1yx, p2yx, p3yx]
    """
    v0 = rp[1] - rp[0]
    v1 = rp[2] - rp[1]
    v2 = rp[3] - rp[2]
    v3 = rp[0] - rp[3]

    return [v0, v1, v2, v3]


def rectangle_homogeneous_lines(rv):
    """

    # Arguments

    rv: rectangle vectors [v0yx, v1yx, v2yx, v3yx]


    # Returns

    [r0abc, r1abc, r2abc, r3abc]

    """
    # ax + by + c = 0
    dv = rv[0] - rv[1]
    # TODO(ahundt) make sure cross product doesn't need to be in xy order
    r0abc = K.concatenate([dv[0], dv[1], tf.cross(rv[0], rv[1])])
    dv = rv[1] - rv[2]
    r1abc = K.concatenate([dv[1], dv[2], tf.cross(rv[1], rv[2])])
    dv = rv[2] - rv[3]
    r2abc = K.concatenate([dv[2], dv[3], tf.cross(rv[2], rv[3])])
    dv = rv[3] - rv[0]
    r3abc = K.concatenate([dv[3], dv[0], tf.cross(rv[3], rv[0])])
    return [r0abc, r1abc, r2abc, r3abc]


def homogeneous_line_intersection(hl0abc, hl1abc):
    """ Given two homogenous lines return the intersection point in y,x coordinates
    """
    a0 = hl0abc[0]
    b0 = hl0abc[1]
    c0 = hl0abc[2]
    a1 = hl1abc[0]
    b1 = hl1abc[1]
    c1 = hl1abc[2]
    w = a0 * b1 - b0 * a1
    py = (c0 * a1 - a0 * c1) / w
    px = (b0 * c1 - c0 * b1) / w
    return [py, px]


def line_at_point(l_abc, p_yx):
    """

    # Arguments

    l_abc: a line in homogenous coodinates
    p_yx: a point with y, x coordinates
    """
    return l_abc[0] * p_yx[1] + l_abc[1] * p_yx[0] + l_abc[2]


def intersection_points(rl0, rp1):
    """ Evaluate rectangle lines at another rectangle's points
    """
    lv = [
        line_at_point(rl0[0], rp1[0]),
        line_at_point(rl0[1], rp1[1]),
        line_at_point(rl0[2], rp1[2]),
        line_at_point(rl0[3], rp1[3]),
    ]
    return lv


def rectangle_intersection_polygon(rp0, rl0, rp1, rl1):
    """ Given two homogenous line rectangles, it returns the points for the polygon representing their intersection.

    # Arguments

    rp0: rectangle 0 defined with points
    rl0: rectangle 0 defined with homogeneous lines
    rp1: rectangle 1 defined with points
    rp1: rectangle 1 defined with homogeneous lines

    # Returns

    [rp0, rp1, rp2, rp3]
    """
    # TODO(ahundt) this function is still set up for eager execution... figure it out as tf calls...
    # http://www.mathopenref.com/coordpolygonarea.html
    # https://stackoverflow.com/a/45268241/99379
    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    for line1 in rl1:
        line_values = [line_at_point(line1, t) for t in rp0]

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the rect0 polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        for s, t, s_value, t_value, line0 in zip(
                rp0,
                rp0[1:] + rp0[:1],
                line_values,
                line_values[1:] + line_values[:1],
                rl0):

            if s_value <= 0:
                new_intersection.append(s)

            st_value = s_value * t_value
            intersection_point = homogeneous_line_intersection(line1, line0)
            if st_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                new_intersection.append(intersection_point)

        intersection = new_intersection


def polygon_area_four_points(rp):
    """
    # Arguments

    rp: polygon defined by 4 points in y,x order
    """
    # partial = p0x * p1y - p0y * p1x
    partial0 = rp[0][1] * rp[1][0] - rp[0][0] * rp[1][1]
    partial1 = rp[1][1] * rp[2][0] - rp[1][0] * rp[2][1]
    partial2 = rp[2][1] * rp[3][0] - rp[2][0] * rp[3][1]
    partial3 = rp[3][1] * rp[0][0] - rp[3][0] * rp[0][1]
    full_sum = partial0 + partial1 + partial2 + partial3
    return 0.5 * full_sum


def sin_cos_height_width_4(height=None, width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/height'], features['bbox/width']]
    else:
        con_cos_height_width = [sin_theta, cos_theta, height, width]
    return K.concatenate(sin_cos_height_width)


def sin_cos_width_3(width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/width']]
    else:
        con_cos_height_width = [sin_theta, cos_theta, width]
    return K.concatenate(sin_cos_height_width)


def sin2_cos2_height_width_4(height=None, width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/height'], features['bbox/width']]
    else:
        con_cos_height_width = [sin_theta, cos_theta, height, width]
    return K.concatenate(sin_cos_height_width)


def sin2_cos2_width_3(width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/sin_theta'], features['bbox/cos_theta'],
                                features['bbox/width']]
    else:
        con_cos_height_width = [sin_theta, cos_theta, width]
    return K.concatenate(sin_cos_height_width)


def approximate_gaussian_ground_truth_image(image_shape, center, grasp_theta, grasp_width, grasp_height, label, sigma_divisor=None):
    """ Gaussian "ground truth" image approximation for a single proposed grasp at a time.

        For use with the Cornell grasping dataset

       see also: ground_truth_images() in cornell_grasp_dataset_writer.py
    """
    if sigma_divisor is None:
        sigma_divisor = FLAGS.sigma_divisor
    grasp_dims = keras.backend.concatenate([grasp_width, grasp_height])
    sigma = keras.backend.max(grasp_dims) / sigma_divisor

    # make sure center value for gaussian is 0.5
    gaussian = gaussian_kernel_2D(image_shape[:2], center=center, sigma=sigma)
    # label 0 is grasp failure, label 1 is grasp success, label 0.5 will have "no effect".
    # gaussian center with label 0 should be subtracting 0.5
    # gaussian center with label 1 should be adding 0.5
    gaussian = ((label * 2) - 1.0) * gaussian
    max_num = K.max(K.max(gaussian), K.placeholder(1.0))
    min_num = K.min(K.min(gaussian), K.placeholder(-1.0))
    gaussian = (gaussian - min_num) / (max_num - min_num)
    return gaussian


def old_batch_inputs(data_files, train, num_epochs, batch_size,
                     num_preprocess_threads, num_readers):
    print(train)
    if train:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=True,
                                                        capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=False,
                                                        capacity=1)

    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
        print('pass')
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        examples_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, examples_serialized = reader.read(filename_queue)

    features = []
    for thread_id in range(num_preprocess_threads):
        feature = parse_and_preprocess(examples_serialized, train, thread_id)

        features.append(feature)

    features = tf.train.batch_join(
        features,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)

    # height = FLAGS.image_size
    # width = FLAGS.image_size
    # depth = 3

    # features['image/decoded'] = tf.reshape(features['image/decoded'], shape=[batch_size, height, width, depth])

    return features


def crop_to_gripper_transform(
        input_image_shape, grasp_center_coordinate, grasp_center_rotation_theta, cropped_image_shape,
        random_translation_max_pixels=None, random_rotation=None, seed=None):
    """ Transform and rotate image to be centered and aligned with proposed grasp.

        Given a gripper center coodinate and rotation angle,
        transform and rotate the image so it is centered with 0 theta.
    """
    transforms = []
    input_image_shape_float = tf.cast(input_image_shape, tf.float32)
    cropped_image_shape = tf.cast(cropped_image_shape, tf.float32)
    half_image_shape = (cropped_image_shape / 2)[:2]

    crop_offset = - grasp_center_coordinate + half_image_shape

    crop_offset = crop_offset[::-1]
    if random_translation_max_pixels is not None:
        crop_offset = crop_offset + rcp.random_translation_offset(random_translation_max_pixels)[:2]
    # reverse yx to xy
    transforms += [tf.contrib.image.translations_to_projective_transforms(crop_offset)]
    input_height_f = cropped_image_shape[0]
    input_width_f = cropped_image_shape[1]

    if random_rotation is not None and random_rotation is not False:
        if isinstance(random_rotation, bool) and random_rotation:
            random_rotation = tf.convert_to_tensor(
                [-math.pi, math.pi], dtype=tf.float32, name='random_theta')
        if isinstance(random_rotation, float):
            random_rotation = tf.convert_to_tensor(
                [-random_rotation, random_rotation], dtype=tf.float32, name='random_theta')
        print('random rotation: ' + str(random_rotation))
        theta = tf.random_uniform([1], minval=random_rotation[0], maxval=random_rotation[1], seed=seed, dtype=tf.float32)
        grasp_center_rotation_theta += theta
    transforms += [tf.contrib.image.angles_to_projective_transforms(
                         grasp_center_rotation_theta, input_height_f, input_width_f)]
    transform = tf.contrib.image.compose_transforms(*transforms)
    # TODO(ahundt) rename features random_* to a more general name, and make the same change in random_crop.py
    features = {
        # TODO(ahundt) should these be positive or negative?
        'random_rotation': -grasp_center_rotation_theta,
        'random_translation_offset': crop_offset,
        'random_projection_transform': transform
    }

    return transform, features


def grasp_success_yx_3(grasp_success=None, cy=None, cx=None, features=None):
    if features is not None:
        combined = [K.cast(features['bbox/grasp_success'], 'float32'),
                    features['bbox/cy'], features['bbox/cx']]
    else:
        combined = [K.cast(grasp_success, 'float32'), cy, cx]
    return K.concatenate(combined)


def parse_and_preprocess(
        examples_serialized, is_training=True, label_features_to_extract=None,
        data_features_to_extract=None, crop_shape=None, output_shape=None,
        random_crop=None, crop_to_gripper=None,
        preprocessing_mode='tf', seed=None, verbose=0):
    """ Parse an example and perform image preprocessing.

    Right now see the code below for the specific feature strings available.

    crop_shape: The shape to which images should be cropped as an intermediate step, this
        affects how much images can be shifted when random crop is used during training.
    output_shape: The shape to which images should be resized after cropping,
        this is also the final output shape.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
    label_features_to_extract: return only specific
        feature strings as the label values, default of
        None returns all features. Specifying
        features reduces overhead and works with
        yield_record() for keras generator compatibility.
    data_features_to_extract: return only specific
        feature strings as the input values,
        default of None returns all features. Specifying
        features reduces overhead and works with
        yield_record() for keras generator compatibility.
    preprocessing_mode: string for the type of channel preprocessing,
       see keras'
       [preprocess_input()](https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py) for details.


    # Returns

        features


      width is space between the gripper plates
     height is range of possible gripper positions along the line
    """
    if crop_shape is None:
        crop_shape = (FLAGS.crop_height, FLAGS.crop_width, 3)
    if output_shape is None:
        output_shape = (FLAGS.resize_height, FLAGS.resize_width)
        output_shape = K.constant(output_shape, 'int32')
    if FLAGS.redundant:
        feature = parse_example_proto_redundant(examples_serialized)
    else:
        feature = parse_example_proto(examples_serialized)

    if random_crop is None:
        random_crop = FLAGS.random_crop
    if crop_to_gripper is None:
        crop_to_gripper = FLAGS.crop_to_gripper

    # TODO(ahundt) clean up, use grasp_dataset.py as reference, possibly refactor to reuse the code
    sensor_image_dimensions = [FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels]
    image_buffer = feature['image/encoded']
    image = tf.image.decode_png(image_buffer, channels=FLAGS.sensor_color_channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, sensor_image_dimensions)
    feature['image/decoded'] = image
    input_image_shape = tf.shape(image)

    # It is critically important if any image augmentation
    # modifies the coordinate system of the image,
    # such as a random crop, rotation etc, the
    # gripper coordinate features must be updated accordingly
    # and consistently, with visualization to ensure it isn't
    # backwards. An example is +theta rotation vs -theta rotation.
    grasp_center_coordinate = K.concatenate([feature['bbox/cy'], feature['bbox/cx']])
    grasp_center_rotation_theta = feature['bbox/theta']
    feature_map['bbox/sin2_theta'] = tf.sin(feature_map['bbox/theta'] * 2.0)
    feature_map['bbox/cos2_theta'] = tf.cos(feature_map['bbox/theta'] * 2.0)
    feature['sin_cos_height_width_4'] = sin_cos_height_width_4(features=feature)
    feature['sin2_cos2_height_width_4'] = sin2_cos2_height_width_4(features=feature)
    # width is space between the gripper plates
    # height is range of possible gripper positions along the line
    feature['sin_cos_width_3'] = sin_cos_width_3(features=feature)
    feature['sin2_cos2_width_3'] = sin2_cos2_width_3(features=feature)

    # perform image augmentation with projective transforms
    # TODO(ahundt) add scaling and use that change to augment width (gripper openness) param
    central_crop = False
    if random_crop and not crop_to_gripper:
        # Note: this option only works well if the crop size is similar to the input size
        transform, random_features = rcp.random_projection_transform(
            K.shape(image), crop_shape, scale=False, rotation=True, translation=True)
    elif not is_training and not crop_to_gripper:
        # simply do a central crop then a resize when not training
        # to match the input without changes
        central_crop = True
        transform = None
    elif crop_to_gripper:
        # default to no rotation
        random_rotation = None
        if FLAGS.crop_to_gripper_and_rotate:
            crop_to_gripper_theta = grasp_center_rotation_theta
            # limit random rotation to 15 degrees
            if is_training and random_crop:
                random_rotation = math.pi / 12
        else:
            crop_to_gripper_theta = K.constant(0, 'float32')
            # allow arbitrary random rotation
            if is_training and random_crop:
                random_rotation = True

        if is_training and random_crop:
            translation_in_box = K.cast(feature['sin_cos_height_width_4'][-2:] // 2, 'int32')
            translation_in_box = tf.minimum(translation_in_box[0], translation_in_box[1])
            # TODO(ahundt) possibly add scale
        else:
            translation_in_box = None

        transform, random_features = crop_to_gripper_transform(
            input_image_shape, grasp_center_coordinate,
            crop_to_gripper_theta, crop_shape,
            random_translation_max_pixels=translation_in_box,
            random_rotation=random_rotation)
    image, preprocessed_grasp_center_coordinate = rcp.transform_crop_and_resize_image(
        image, crop_shape=crop_shape, resize_shape=output_shape, central_crop=central_crop,
        transform=transform, coordinate=grasp_center_coordinate)

    if 'random_rotation' in random_features:
        # TODO(ahundt) validate if we must subtract or add based on the transform
        grasp_center_rotation_theta += random_features['random_rotation']
    feature.update(random_features)

    feature['image/transformed'] = image

    # perform color augmentation and scaling
    image = preprocess_image(
        image, is_training=is_training, fast_mode=False,
        lower=0.5, upper=1.5, hue_max_delta=0.2,
        brightness_max_delta=32. / 255., mode=preprocessing_mode)

    # generate all the preprocessed features for training
    feature['image/preprocessed'] = image
    feature['image/preprocessed/height'] = K.shape(image)[0]
    feature['image/preprocessed/width'] = K.shape(image)[1]

    feature['bbox/preprocessed/cy'] = preprocessed_grasp_center_coordinate[0]
    feature['bbox/preprocessed/cx'] = preprocessed_grasp_center_coordinate[1]
    feature['bbox/preprocessed/cy_cx_normalized_2'] = K.concatenate(
        [tf.reshape(preprocessed_grasp_center_coordinate[0], (1,)) / tf.cast(feature['image/preprocessed/height'], tf.float32),
         tf.reshape(preprocessed_grasp_center_coordinate[1], (1,)) / tf.cast(feature['image/preprocessed/width'], tf.float32)])
    feature['bbox/preprocessed/theta'] = grasp_center_rotation_theta
    feature['bbox/preprocessed/sin_cos_2'] = K.concatenate(
        [K.sin(grasp_center_rotation_theta), K.cos(grasp_center_rotation_theta)])
    random_scale = K.constant(1.0, 'float32')
    if 'random_scale' in feature:
        random_scale = feature['random_scale']
    feature['bbox/preprocessed/height'] = feature['bbox/height'] * random_scale
    feature['bbox/preprocessed/width'] = feature['bbox/width'] * random_scale
    feature['bbox/preprocessed/logarithm_height_width_2'] = K.concatenate(
        [K.log(feature['bbox/preprocessed/height'] + K.epsilon()),
         K.log(feature['bbox/preprocessed/width'] + K.epsilon())])
    # TODO(ahundt) difference between "redundant" and regular proto parsing, figure out how to deal with grasp_success rename properly
    feature['grasp_success'] = feature['bbox/grasp_success']
    grasp_success_coordinate_label = K.concatenate(
        [tf.cast(feature['bbox/grasp_success'], tf.float32), grasp_center_coordinate])
    # make coordinate labels 4d because that's what keras expects
    grasp_success_coordinate_label = K.expand_dims(K.expand_dims(grasp_success_coordinate_label))
    feature['grasp_success_yx_3'] = grasp_success_coordinate_label

    # TODO(ahundt) reenable this and compare performance against segmentation_gaussian_measurement()
    if False:
        feature['grasp_success_2D'] = approximate_gaussian_ground_truth_image(
            image_shape=keras.backend.int_shape(image),
            center=[feature['bbox/cy'], feature['bbox/cx']],
            grasp_theta=feature['bbox/theta'],
            grasp_width=feature['bbox/width'],
            grasp_height=feature['bbox/height'],
            label=feature['grasp_success'])

    if verbose > 0:
        print(feature)

    if label_features_to_extract is None:
        return feature
    else:
        # strip out all features that aren't needed to reduce processing time
        simplified_feature = {}
        for feature_name in data_features_to_extract:
            simplified_feature[feature_name] = feature[feature_name]
        for feature_name in label_features_to_extract:
            simplified_feature[feature_name] = feature[feature_name]
        return simplified_feature


def yield_record(
        tfrecord_filenames, label_features_to_extract=None, data_features_to_extract=None,
        parse_example_proto_fn=parse_and_preprocess, batch_size=32,
        device='/cpu:0', is_training=True, steps=None, buffer_size=int(1e6),
        shuffle=True, shuffle_buffer_size=100, num_parallel_calls=None, preprocessing_mode='tf'):
    if num_parallel_calls is None:
        num_parallel_calls = FLAGS.num_readers
    # based_on https://github.com/visipedia/tfrecords/blob/master/iterate_tfrecords.py
    # with tf.device(device):
    with tf.Session() as sess:

        dataset = tf.data.TFRecordDataset(
            tfrecord_filenames, buffer_size=buffer_size)
        dataset = dataset.repeat(count=steps)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        # Repeat the input indefinitely.

        # call the parse_example_proto_fn with the is_training flag set.
        def parse_fn_is_training(example):
            return parse_example_proto_fn(examples_serialized=example, is_training=is_training,
                                          label_features_to_extract=label_features_to_extract,
                                          data_features_to_extract=data_features_to_extract,
                                          preprocessing_mode=preprocessing_mode)
        dataset = dataset.map(
            map_func=parse_fn_is_training,
            num_parallel_calls=num_parallel_calls)
        if batch_size > 1:
            dataset = dataset.batch(batch_size=batch_size)  # Parse the record into tensors.
        dataset = dataset.prefetch(batch_size * 5)
        tensor_iterator = dataset.make_one_shot_iterator()
        #     # Construct a Reader to read examples from the .tfrecords file
        #     reader = tf.TFRecordReader()
        #     _, serialized_example = reader.read(filename_queue)

        #     features = decode_serialized_example(serialized_example, features_to_extract)

        # coord = tf.train.Coordinator()
        # sess = K.get_session()
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        # tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            # while not coord.should_stop():
            # features = sess.run(tensor_iterator.initializer)
            # If `Iterator.get_next()` is being called inside a training loop,
            # which will cause gradual slowdown and eventual resource exhaustion.
            # If this is the case, restructure your code to call `next_element = iterator.get_next()
            # once outside the loop, and use `next_element` inside the loop.
            next_element = tensor_iterator.get_next()
            while True:
                # print('start iteration')
                features = sess.run(next_element)
                # print('run complete')

                # extract the features in a specific order if
                # features_to_extract is specified,
                # useful for yielding to a generator
                # like with keras.model.fit().
                # outputs should be the outputs of the
                # dataset API, consisting of two lists:
                # [[inputs], [labels]]
                # "labels" is called "targets" by keras
                #
                # See documentation for keras.model.fit_generator()
                # https://keras.io/models/model/
                #
                # print('start list_processing')
                if data_features_to_extract is not None and label_features_to_extract is not None:
                    outputs = ([features[feature_name] for feature_name in data_features_to_extract],
                               [features[feature_name] for feature_name in label_features_to_extract])
                # print('done list proc, about to yield')
                    yield outputs
                else:
                    yield features
                #

        except tf.errors.OutOfRangeError as e:
            pass


def old_distorted_inputs(data_files, num_epochs, train=True, batch_size=None):
    with tf.device('/cpu:0'):
        print(train)
        features = old_batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)

    return features


def old_inputs(data_files, num_epochs=1, train=False, batch_size=1):
    with tf.device('/cpu:0'):
        print(train)
        features = old_batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=1)

    return features


def visualize_redundant_example(features_dicts, showTextBox=None):
    """ Visualize numpy dictionary containing a grasp example.
    """
    if showTextBox is None:
        showTextBox = FLAGS.showTextBox
    # TODO(ahundt) don't duplicate this in cornell_grasp_dataset_writer
    if not isinstance(features_dicts, list):
        features_dicts = [features_dicts]
    width = 3

    preprocessed_examples = []
    for example in features_dicts:
        print('original example bbox/theta: ' + str(example['bbox/theta']))
        if ('bbox/preprocessed/cy_cx_normalized_2' in example):
            # Reverse the preprocessing so we can visually compare correctness
            decoded_example = copy.deepcopy(example)
            sin_cos_2 = example['bbox/preprocessed/sin_cos_2']
            # y, x ordering.
            recovered_theta = np.arctan2(sin_cos_2[0], sin_cos_2[1])
            if 'random_rotation' in example:
                recovered_theta += example['random_rotation']
            cy_cx_normalized_2 = example['bbox/preprocessed/cy_cx_normalized_2']
            cy_cx_normalized_2[0] *= example['image/preprocessed/height']
            cy_cx_normalized_2[1] *= example['image/preprocessed/width']
            if 'random_translation_offset' in example:
                offset = example['random_translation_offset']
                print('offset: ' + str(offset))
            # if np.allclose(np.array(example['bbox/theta']), recovered_theta):
            #     print('WARNING: bbox/theta: ' + str(example['bbox/theta']) +
            #           ' feature does not match bbox/preprocessed/sin_cos_2: '
            #           )
            # # change to preprocessed
            # assert np.allclose(cy_cx_normalized_2[0] + offset[0], example['bbox/cy'])
            # assert np.allclose(cy_cx_normalized_2[1] + offset[1], example['bbox/cx'])
            decoded_example['bbox/theta'] = example['bbox/preprocessed/theta']
            decoded_example['image/decoded'] = example['image/preprocessed']
            decoded_example['bbox/width'] = example['bbox/preprocessed/width']
            decoded_example['bbox/height'] = example['bbox/preprocessed/height']
            decoded_example['bbox/cy'] = example['bbox/preprocessed/cy']
            decoded_example['bbox/cx'] = example['bbox/preprocessed/cx']
            if 'random_projection_transform' in example:
                print('random_projection_transform:' + str(example['random_projection_transform']))
                if 'random_rotation' in example:
                    print('random_rotation: ' + str(example['random_rotation']))
                print('bbox/preprocessed/theta: ' + str(example['bbox/preprocessed/theta']))
            preprocessed_examples.append(decoded_example)

    img = example['image/decoded']
    center_x_list = [example['bbox/cx'] for example in features_dicts]
    center_y_list = [example['bbox/cy'] for example in features_dicts]
    grasp_success = [example['bbox/grasp_success'] for example in features_dicts]
    gt_plot_height = int(np.ceil(float(len(center_x_list)) / 2))
    fig, axs = plt.subplots(gt_plot_height + 1, 4, figsize=(15, 15))
    print('max: ' + str(np.max(img)) + ' min: ' + str(np.min(img)))
    axs[0, 0].imshow(img, zorder=0)
    # for i in range(4):
    #     feature['bbox/y' + str(i)] = _floats_feature(dict_bbox_lists['bbox/y' + str(i)])
    #     feature['bbox/x' + str(i)] = _floats_feature(dict_bbox_lists['bbox/x' + str(i)])
    # axs[0, 0].arrow(np.array(center_y_list), np.array(center_x_list),
    #                 np.array(coordinates_list[0]) - np.array(coordinates_list[2]),
    #                 np.array(coordinates_list[1]) - np.array(coordinates_list[3]), c=grasp_success)
    axs[0, 0].scatter(np.array(center_x_list), np.array(center_y_list), zorder=2, c=grasp_success, alpha=0.5, lw=2)
    axs[0, 1].imshow(img, zorder=0)
    # plt.show()
    # axs[1, 0].scatter(data[0], data[1])
    # axs[2, 0].imshow(gt_image)
    for i, example in enumerate(preprocessed_examples):
        h = i % gt_plot_height + 1
        w = int(i / gt_plot_height)
        z = 0
        # axs[h, w].imshow(img, zorder=z)
        z += 1

        img2 = example['image/decoded']
        # Assuming 'tf' preprocessing mode! Changing channel range from [-1, 1] to [0, 1]
        img2 /= 2
        img2 += 0.5
        print('preprocessed max: ' + str(np.max(img2)) + ' min: ' + str(np.min(img2)) + ' shape: ' + str(np.shape(img2)))
        axs[h, w].imshow(img2, alpha=1, zorder=z)
        # plt.show()
        print('did I make it?')
        z += 1
        # axs[h, w*2+1].imshow(gt_image, alpha=0.75, zorder=1)
        widths = [1, 2, 1, 2]
        alphas = [0.25, 0.5, 0.25, 0.5]
        if example['bbox/grasp_success']:
            # that's gap, plate, gap plate
            colors = ['gray', 'green', 'gray', 'green']
            success_str = 'pos'
        else:
            colors = ['gray', 'purple', 'gray', 'purple']
            success_str = 'neg'

        cx = [int(example['bbox/cx'])]
        cy = [int(example['bbox/cy'])]
        if showTextBox:
            axs[h, w].text(
                    int(cx[0]), int(cy[0]),
                    success_str, size=10,
                    # see the gripper angle
                    rotation=float(np.rad2deg(example['bbox/theta'])),
                    # this next one is just to see if random rotation matches actual image rotation
                    # rotation=float(np.rad2deg(example['random_rotation'])),
                    ha="right", va="top",
                    bbox=dict(boxstyle="square",
                              ec=(1., 0.5, 0.5),
                              fc=(1., 0.8, 0.8),
                              ),
                    zorder=z,
                    )
            z += 1

        axs[h, w].scatter(cx, cy, zorder=2, alpha=0.5, lw=2)
        # bbox/x_i, y_i are not calculated according to preprocess, so use rectangle here
        # axs[h, w].add_patch(patches.Rectangle((example['bbox/cx'], example['bbox/cy']),
        #                                       example['bbox/width'], example['bbox/height'],
        #                                       example['bbox/theta'], fill=False))
        # for i, (color, width, alpha) in enumerate(zip(colors, widths, alphas)):
        #     x_current = [example['bbox/x'+str(i)], example['bbox/x'+str((i+1)%4)]]
        #     y_current = [example['bbox/y'+str(i)], example['bbox/y'+str((i+1)%4)]]
        #     # axs[h, w].text(example['bbox/x'+str(i)], example['bbox/y'+str(i)], "Point:"+str(i))

        #     axs[h, w].add_line(lines.Line2D(x_current, y_current, linewidth=width,
        #                        color=color, zorder=z, alpha=alpha))
        #     axs[0, 0].add_line(lines.Line2D(x_current, y_current, linewidth=width,
        #                        color=color, zorder=z, alpha=alpha))
        plt.show()

        print('did I make it2?')
    # axs[1, 1].hist2d(data[0], data[1])
    # plt.draw()
    # plt.pause(0.25)

    plt.show()
    return width


def main(argv):
    batch_size = 1
    is_training = False
    validation_file = FLAGS.evaluate_filename

    for example_dict in tqdm(yield_record(
            validation_file, is_training=is_training,
            batch_size=batch_size)):
        visualize_redundant_example(example_dict, showTextBox=True)


if __name__ == '__main__':
    tf.app.run(main=main)
