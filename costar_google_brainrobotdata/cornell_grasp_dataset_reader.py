#!/usr/local/bin/python
'''
Image processing of cornell grasping dataset for detecting grasping positions.

Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0

Cornell Dataset Code based on:
    https://github.com/tnikolla/robot-grasp-detection

'''
import os
import copy
import numpy as np

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
flags.DEFINE_integer('crop_width', 480,
                     """Width to crop images""")
flags.DEFINE_integer('crop_height', 360,
                     """Height to crop images""")
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
                     """)
flags.DEFINE_integer('resize_width', 320,
                     """Width to resize images before prediction, if enabled.""")
flags.DEFINE_integer('resize_height', 240,
                     """Height to resize images before prediction, if enabled.""")
flags.DEFINE_boolean('resize', True,
                     """resize will resize the input images to the desired dimensions specified by the
                        resize_width and resize_height flags. It is suggested that an exact factor of 2 be used
                        relative to the input image directions if random_crop is disabled or the crop dimensions otherwise.
                     """)

FLAGS = flags.FLAGS


def parse_example_proto(examples_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64)}
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

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def parse_example_proto_redundant(examples_serialized):
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


def grasp_success_yx_3(grasp_success=None, cy=None, cx=None, features=None):
    if features is not None:
        combined = [K.cast(features['bbox/grasp_success'], 'float32'),
                    features['bbox/cy'], features['bbox/cx']]
    else:
        combined = [K.cast(grasp_success, 'float32'), cy, cx]
    return K.concatenate(combined)


def parse_and_preprocess(examples_serialized, is_training=True, label_features_to_extract=None,
                         data_features_to_extract=None, crop_shape=None, output_shape=None,
                         preprocessing_mode='tf'):
    """
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
    if is_training:
        # perform image augmentation
        # TODO(ahundt) add scaling and use that change to augment height and width parameters
        transform, random_features = rcp.random_projection_transform(K.shape(image), crop_shape)
        image, grasp_center_coordinate = rcp.transform_crop_and_resize_image(
            image, crop_shape=crop_shape, resize_shape=output_shape,
            transform=transform, coordinate=grasp_center_coordinate)

        if 'random_rotation' in random_features:
            # TODO(ahundt) validate if we must subtract or add based on the transform
            grasp_center_rotation_theta += random_features['random_rotation']
        feature.update(random_features)
    else:
        # simply do a central crop then a resize when not training
        # to match the input without  changes
        image, grasp_center_coordinate = rcp.transform_crop_and_resize_image(
            image, crop_shape=crop_shape, central_crop=True,
            resize_shape=output_shape, coordinate=grasp_center_coordinate)

    # perform color augmentation and scaling
    image = preprocess_image(image, is_training=is_training, mode=preprocessing_mode)

    # generate all the preprocessed features for training
    feature['image/preprocessed'] = image
    feature['bbox/preprocessed/cy_cx_2'] = grasp_center_coordinate
    feature['bbox/preprocessed/cy_cx_normalized_2'] = K.concatenate(
        [tf.reshape(grasp_center_coordinate[0], (1,)) / tf.cast(feature['image/height'], tf.float32),
         tf.reshape(grasp_center_coordinate[1], (1,)) / tf.cast(feature['image/width'], tf.float32)])
    feature['bbox/preprocessed/theta'] = grasp_center_rotation_theta
    feature['bbox/preprocessed/sin_cos_2'] = K.concatenate(
        [K.sin(grasp_center_rotation_theta), K.cos(grasp_center_rotation_theta)])
    feature['bbox/preprocessed/logarithm_height_width_2'] = K.concatenate(
        [K.log(feature['bbox/height'] + K.epsilon()),
         K.log(feature['bbox/width'] + K.epsilon())])

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
        tfrecord_filenames, label_features_to_extract, data_features_to_extract,
        parse_example_proto_fn=parse_and_preprocess, batch_size=32,
        device='/cpu:0', is_training=True, steps=None, buffer_size=int(4e8),
        num_parallel_calls=FLAGS.num_readers, preprocessing_mode='tf'):
    # based_on https://github.com/visipedia/tfrecords/blob/master/iterate_tfrecords.py
    # with tf.device(device):
    with tf.Session() as sess:

        dataset = tf.data.TFRecordDataset(
            tfrecord_filenames, buffer_size=buffer_size)
        dataset = dataset.repeat(count=steps)
        # Repeat the input indefinitely.

        # call the parse_example_proto_fn with the is_training flag set.
        def parse_fn_is_training(example):
            return parse_example_proto_fn(examples_serialized=example, is_training=is_training,
                                          label_features_to_extract=label_features_to_extract,
                                          data_features_to_extract=data_features_to_extract,
                                          preprocessing_mode=preprocessing_mode)
        dataset = dataset.map(
            map_func=parse_and_preprocess,
            num_parallel_calls=num_parallel_calls)
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
                outputs = ([features[feature_name] for feature_name in data_features_to_extract],
                           [features[feature_name] for feature_name in label_features_to_extract])
                # print('done list proc, about to yield')
                yield outputs
                #

        except tf.errors.OutOfRangeError as e:
            pass


def old_distorted_inputs(data_files, num_epochs, train=True, batch_size=None):
    with tf.device('/cpu:0'):
        print(train)
        features = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)

    return features


def old_inputs(data_files, num_epochs=1, train=False, batch_size=1):
    with tf.device('/cpu:0'):
        print(train)
        features = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=1)

    return features


def visualize_redundant_example(features_dicts, showTextBox=FLAGS.showTextBox):
    """ Visualize numpy dictionary containing a grasp example.
    """
    # TODO(ahundt) don't duplicate this in cornell_grasp_dataset_writer
    if not isinstance(features_dicts, list):
        features_dicts = [features_dicts]
    width = 3

    preprocessed_examples = []
    for example in features_dicts:
        if ('bbox/preprocessed/cy_cx_2' in example):
            sin_cos_2 = example['bbox/preprocessed/sin_cos_2']
            decoded_example = copy.deepcopy(example)
            # y, x ordering.
            recovered_theta = np.atan2(sin_cos_2[0], sin_cos_2[1])
            assert np.allclose(np.array(example['bbox/theta']), recovered_theta)
            decoded_example['bbox/theta'] = recovered_theta

            cy_cx_normalized_2 = example['bbox/preprocessed/cy_cx_normalized_2']
            bbox



    center_x_list = [example['bbox/cx'] for example in features_dict]
    center_y_list = [example['bbox/cy'] for example in features_dict]
    grasp_success = [example['bbox/grasp_success'] for example in features_dict]
    gt_plot_height = len(center_x_list)/2
    fig, axs = plt.subplots(gt_plot_height + 1, 4, figsize=(15, 15))
    axs[0, 0].imshow(img, zorder=0)
    # for i in range(4):
    #     feature['bbox/y' + str(i)] = _floats_feature(dict_bbox_lists['bbox/y' + str(i)])
    #     feature['bbox/x' + str(i)] = _floats_feature(dict_bbox_lists['bbox/x' + str(i)])
    # axs[0, 0].arrow(np.array(center_y_list), np.array(center_x_list),
    #                 np.array(coordinates_list[0]) - np.array(coordinates_list[2]),
    #                 np.array(coordinates_list[1]) - np.array(coordinates_list[3]), c=grasp_success)
    axs[0, 0].scatter(np.array(center_x_list), np.array(center_y_list), zorder=2, c=grasp_success, alpha=0.5, lw=2)
    axs[0, 1].imshow(img, zorder=0)
    # axs[1, 0].scatter(data[0], data[1])
    # axs[2, 0].imshow(gt_image)
    for i, (gt_image, example) in enumerate(zip(gt_images, bbox_example_features)):
        h = i % gt_plot_height + 1
        w = int(i / gt_plot_height)
        z = 0
        axs[h, w].imshow(img, zorder=z)
        z += 1
        axs[h, w].imshow(gt_image, alpha=0.25, zorder=z)
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

        if showTextBox:
            axs[h, w].text(
                    example['bbox/cx'], example['bbox/cy'],
                    success_str, size=10, rotation=-np.rad2deg(example['bbox/theta']),
                    ha="right", va="top",
                    bbox=dict(boxstyle="square",
                              ec=(1., 0.5, 0.5),
                              fc=(1., 0.8, 0.8),
                              ),
                    zorder=z,
                    )
            z += 1
        for i, (color, width, alpha) in enumerate(zip(colors, widths, alphas)):
            x_current = [example['bbox/x'+str(i)], example['bbox/x'+str((i+1)%4)]]
            y_current = [example['bbox/y'+str(i)], example['bbox/y'+str((i+1)%4)]]
            # axs[h, w].text(example['bbox/x'+str(i)], example['bbox/y'+str(i)], "Point:"+str(i))

            axs[h, w].add_line(lines.Line2D(x_current, y_current, linewidth=width,
                               color=color, zorder=z, alpha=alpha))
            axs[0, 0].add_line(lines.Line2D(x_current, y_current, linewidth=width,
                               color=color, zorder=z, alpha=alpha))

    # axs[1, 1].hist2d(data[0], data[1])
    plt.draw()
    plt.pause(0.25)

    plt.show()
    return width


def main(batch_size=1, is_training=True):
    example_generator = cornell_grasp_dataset_reader.yield_record(
        validation_file, label_features, data_features,
        is_training=is_training, batch_size=batch_size,
        parse_example_proto_fn=parse_and_preprocess)

    for example_dict in tqdm(example_generator):
        visualize_example(example_dict)




if __name__ == '__main__':
    main()