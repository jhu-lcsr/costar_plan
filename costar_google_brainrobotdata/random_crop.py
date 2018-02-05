import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops


def random_crop_offset(input_shape, output_shape, seed=None, name=None):
    """ Generate a random image corner offset to randomly crop an image.

        The return value should be supplied to crop_images().

        # Arguments

        input_shape: shape of input image to crop.
        output_shape: shape of image after cropping.
        seed: Python integer. Used to create a random seed.
            See tf.set_random_seed for behavior.
        name: A name for this operation.
    """
    with tf.name_scope(name, "random_crop",
                       [input_shape, output_shape]) as name:
        input_shape = ops.convert_to_tensor(
            input_shape, dtype=dtypes.int32, name="input_shape")
        output_shape = ops.convert_to_tensor(
            output_shape, dtype=dtypes.int32, name="output_shape")
        check = control_flow_ops.Assert(
            math_ops.reduce_all(input_shape >= output_shape),
            ["Need input_shape >= output_shape ", input_shape, output_shape],
            summarize=1000)
        input_shape = control_flow_ops.with_dependencies([check], input_shape)
        limit = input_shape - output_shape + 1
        offset = tf.random_uniform(
            array_ops.shape(input_shape),
            dtype=output_shape.dtype,
            maxval=output_shape.dtype.max,
            seed=seed) % limit

    return offset


def crop_images(image_list, offset, size, name=None, verbose=0):
    """ Crop color image and depth image to specified offset and size.

        random_crop_offset() can be utilized to generate the offset parameter.
        Any associated image intrinsics matrix, can be updated with crop_image_intrinsics().

        To update an image coordinate after the crop simply do:

            new_image_coordinate = original_image_coordinate - offset

        # Arguments

        offset: The coordinate offset from the origin at which the cropped image should begin.
        image_list: input images need to crop.
        image_intrinsics: 3*3 matrix with focal lengths, and principle points,
        size: output size of image after cropping, must be less than or equal to the image size.
        name: A name for this operation.
    """
    with tf.name_scope(name, "crop_images", [image_list, size]) as name:
        if isinstance(image_list, list):
            cropped_image_list = []
            size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
            for image in image_list:
                image = ops.convert_to_tensor(image, name="image")
                image = array_ops.slice(image, offset, size, name=name)
                cropped_image_list.append(image)
        else:
            size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
            image_list = ops.convert_to_tensor(image_list, name="image")
            offset = ops.convert_to_tensor(offset, dtype=dtypes.int32, name="offset")
            if verbose > 0:
                print("crop_images offset:", offset, 'size', size, 'img_list_shape', image_list.shape)
                offset = tf.Print(offset, [offset, size, image_list.shape])
            cropped_image_list = array_ops.slice(
                image_list, offset, size, name=name)

    return cropped_image_list


def crop_image_intrinsics(camera_intrinsics_matrix, offset, name=None):
    """ Calculate an updated image intrinsics matrix after a cropping with the specified offset.

        # Arguments

        camera_intrinsics_matrix: intrinsic matrix before cropping.
        offset: offset used in cropping.
    """
    # offset should be array can be access by index
    with tf.name_scope(name, "crop_image_intrinsics", [camera_intrinsics_matrix, offset]) as name:
        offset = tf.cast(offset, camera_intrinsics_matrix.dtype)
        offset_matrix = tf.convert_to_tensor(
            [[0., 0., 0.],
             [0., 0., 0.],
             [offset[0], offset[1], 0.]])
        return camera_intrinsics_matrix - offset_matrix


def random_projection_transform(
        input_shape=None, output_shape=None,
        translation=True, scale=False, rotation=True,
        horizontal_flip=True, vertical_flip=True,
        name=None, seed=None):
    """ Generate a random projection data augmentation transform.

    Please note this is not truly random in SE(3), and shear is not yet supported.

    # Arguments

        input_shape: The shape of the image to which the projection will be applied.
        scale: False means no scaling will be applied, True calculates the range of scaling
            to be +/- the change in scale from the input shape to the output shape.
            Range of valid scaling multiples, for example (0.5, 1.5)
            is 2x zoom in to 1.5x zoom out the Default None is (1.0, 1.0).
        rotation: Range of valid random rotation angles. The default None is (0.0, 0.0)
        required_coordinate: A coordinate which must be included in the output image.
        crop_size: The final dimensions expected for the output image
            a random translation will be created so you can make a central
            crop to the crop size. The crop size must be equal to or
            smaller than the input size.
        seed: A random seed.

    # Returns

        An 8x1 random projection transform.
    """
    with tf.name_scope(name, "random_projection",
                       [input_shape, output_shape]) as name:
        input_shape = ops.convert_to_tensor(
            input_shape, dtype=dtypes.int32, name="input_shape")
        if output_shape is None:
            output_shape = input_shape
        output_shape = ops.convert_to_tensor(
            output_shape, dtype=dtypes.int32, name="output_shape")
        check = control_flow_ops.Assert(
            math_ops.reduce_all(input_shape >= output_shape),
            ["Need input_shape >= output_shape ", input_shape, output_shape],
            summarize=1000)
        input_shape = control_flow_ops.with_dependencies([check], input_shape)

        transforms = []
        input_height_f = tf.cast(input_shape[0], tf.float32)
        input_width_f = tf.cast(input_shape[1], tf.float32)

        if translation is not None and translation is not False:
            if isinstance(translation, bool) and translation:
                offset = random_crop_offset(input_shape, output_shape, seed=seed)
            else:
                offset = translation
            # The transform is float32, which differs from
            # random_crop_offset which is int32
            offset = tf.cast(offset, tf.float32)
        else:
            offset = (0., 0.)

        # there should always be some offset, even if it is (0, 0)
        transforms += [tf.contrib.image.translations_to_projective_transforms(offset)]
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation is not None and rotation is not False:
            if isinstance(rotation, bool) and rotation:
                rotation = ops.convert_to_tensor(
                    [-math.pi, math.pi], dtype=tf.float32, name="input_shape")

            theta = tf.random_uniform([1], minval=rotation[0], maxval=rotation[1], seed=seed, dtype=tf.float32)
            transforms += [tf.contrib.image.angles_to_projective_transforms(theta, input_height_f, input_width_f)]

        if scale is not None and scale is not False:
            if isinstance(scale, bool) and scale:
                # choose the scale from the input to output image size difference
                max_input_dim = input_shape[tf.argmax(input_shape)]
                max_crop_dim = output_shape[tf.argmax(output_shape)]
                s0 = tf.cast(input_shape[max_input_dim], dtype=tf.float32) / tf.cast(max_crop_dim, dtype=tf.float32)
                s1 = tf.cast(output_shape[max_crop_dim], dtype=tf.float32) / tf.cast(max_input_dim, dtype=tf.float32)
                scale = [s0, s1]

            s = tf.random_uniform([1], minval=scale[0], maxval=scale[1], seed=seed, dtype=tf.float32)
            scale_matrix = [s,  array_ops.zeros((1), dtypes.float32), array_ops.zeros((1), dtypes.float32),
                            array_ops.zeros((1), dtypes.float32), s, array_ops.zeros((1), dtypes.float32),
                            array_ops.zeros((1), dtypes.float32), array_ops.zeros((1), dtypes.float32)]
            scale_matrix = tf.stack(scale_matrix, axis=-1)
            transforms += [scale_matrix]

        batch_size = 1
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., input_width_f, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., input_height_f, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        composed_transforms = tf.contrib.image.compose_transforms(*transforms)
        return composed_transforms


def transform_and_crop_coordinate(coordinate, transform=None, offset=None):
    """ Transforms a single coordinate then applies a crop offset.

     See transform_and_crop().

    # Arguments

        coordinate: A 2D image coordinate.
        transform: A 3x3 homogenous 2D image transform matrix.
        offset: A crop offset.
    """
    projection_matrix = tf.contrib.image._flat_transforms_to_matrices(transform)
    if transform is not None:
        coordinate = tf.transpose(tf.convert_to_tensor(
            coordinate[0],
            coordinate[1],
            1
        ))
        coordinate = projection_matrix * coordinate
        coordinate = coordinate[:2]
    if offset is not None:
        coordinate = coordinate - offset
    return coordinate


def transform_and_crop_image(image, offset=None, size=None, transform=None, interpolation='BILINEAR', coordinate=None):
    """ Project the image with a 3x3 htransform and center crop the image to the output shape.

    For converting between transformation matrices and homogeneous transforms see:
    transform_and_crop_coordinate()

    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/image/python/ops/image_ops.py

    tf.contrib.image._flat_transforms_to_matrices()
    tf.contrib.image._transform_matrices_to_flat()

    Please note this function does not yet have an equivalent to crop_image_intrinsics.

    # Arguments
        offset: an offset to perform after the transform is applied,
           if not defined it defaults to central crop which is half the
           size difference between the input image and offset.
        size: The output image size, default None is the input image size.
        transform: An 8 element homogeneous projective transformation matrix.
        interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".

    # Returns

       cropped_image if coordinate is None, otherwise [cropped_image, new_coordinate]
    """
    if size is None and offset is not None:
        raise ValueError('If size is None offset must also be None.')

    image = tf.contrib.image.transform(image, transforms=transform, interpolation=interpolation)

    if size is not None or offset is not None:
        if size is None:
            size = tf.shape(image)

        if offset is None:
            offset = (tf.shape(image) - size) // 2

        image = crop_images(image, offset, size)

    if coordinate is None:
        return image
    else:
        return image, transform_and_crop_coordinate(transform, coordinate, offset)

# def _flat_transforms_to_matrices(transforms):
#   # Make the transform(s) 2D in case the input is a single transform.
#   transforms = array_ops.reshape(transforms, constant_op.constant([-1, 8]))
#   num_transforms = array_ops.shape(transforms)[0]
#   # Add a column of ones for the implicit last entry in the matrix.
#   return array_ops.reshape(
#       array_ops.concat(
#           [transforms, array_ops.ones([num_transforms, 1])], axis=1),
#       constant_op.constant([-1, 3, 3]))


# def _transform_matrices_to_flat(transform_matrices):
#   # Flatten each matrix.
#   transforms = array_ops.reshape(transform_matrices,
#                                  constant_op.constant([-1, 9]))
#   # Divide each matrix by the last entry (normally 1).
#   transforms /= transforms[:, 8:9]
#   return transforms[:, :8]



