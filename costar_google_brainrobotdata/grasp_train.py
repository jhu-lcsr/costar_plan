import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
import grasp_dataset
import grasp_model

from tensorflow.python.platform import flags


tf.flags.DEFINE_string('grasp_model', 'grasp_model_single',
                       """Choose the model definition to run, options are grasp_model and grasp_model_segmentation""")
tf.flags.DEFINE_string('save_weights', 'grasp_model_weights.h5',
                       """Save a file with the trained model weights.""")
tf.flags.DEFINE_string('load_weights', 'grasp_model_weights.h5',
                       """Load and continue training the specified file containing model weights.""")
tf.flags.DEFINE_integer('epochs', 100,
                        """Epochs of training""")
tf.flags.DEFINE_integer('random_crop_width', 472,
                        """Width to randomly crop images, if enabled""")
tf.flags.DEFINE_integer('random_crop_height', 472,
                        """Height to randomly crop images, if enabled""")
tf.flags.DEFINE_boolean('random_crop', False,
                        """NOT YET SUPPORTED. random_crop will apply the tf random crop function with
                           the parameters specified by random_crop_width and random_crop_height
                        """)
tf.flags.DEFINE_boolean('image_augmentation', True,
                        'image augmentation applies random brightness, saturation, hue, contrast')
tf.flags.DEFINE_boolean('imagenet_mean_subtraction', True,
                        'subtract the imagenet mean pixel values from the rgb images')
# tf.flags.DEFINE_integer('batch_size', 1,
#                         """size of a single batch during training""")

FLAGS = flags.FLAGS


class GraspTrain(object):

    def _image_augmentation(image):
        """Performs data augmentation by randomly permuting the inputs.

        Source: https://github.com/tensorflow/models/blob/aed6922fe2da5325bda760650b5dc3933b10a3a2/domain_adaptation/pixel_domain_adaptation/pixelda_preprocess.py#L81

        Args:
            image: A float `Tensor` of size [height, width, channels] with values
            in range[0,1].
        Returns:
            The mutated batch of images
        """
        # Apply photometric data augmentation (contrast etc.)
        num_channels = image.shape_as_list()[-1]
        if num_channels == 4:
            # Only augment image part
            image, depth = image[:, :, 0:3], image[:, :, 3:4]
        elif num_channels == 1:
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0, 1.0)
        if num_channels == 4:
            image = tf.concat(2, [image, depth])
        elif num_channels == 1:
            image = tf.image.rgb_to_grayscale(image)
        return image

    def _imagenet_mean_subtraction(self, tensor):
        """Do imagenet preprocessing, but make sure the network you are using needs it!

           zero centers by mean pixel.
        """
        # TODO(ahundt) do we need to divide by 255 to make it floats from 0 to 1? It seems no based on https://keras.io/applications/
        # TODO(ahundt) apply resolution to https://github.com/fchollet/keras/pull/7705 when linked PR is closed
        # TODO(ahundt) also apply per image standardization?
        pixel_value_offset = tf.constant([103.939, 116.779, 123.68])
        return tf.subtract(tensor, pixel_value_offset)

    def _rgb_preprocessing(self, rgb_image_op,
                           image_augmentation=FLAGS.image_augmentation,
                           imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction):
        """Preprocess an rgb image into a float image, applying image augmentation and imagenet mean subtraction if desired
        """
        # make sure the shape is correct
        rgb_image_op = tf.squeeze(rgb_image_op)
        # apply image augmentation and imagenet preprocessing steps adapted from keras
        if image_augmentation:
            rgb_image_op = self._image_augmentation(rgb_image_op)
        rgb_image_op = tf.cast(rgb_image_op, tf.float32)
        if imagenet_mean_subtraction:
            rgb_image_op = self._imagenet_mean_subtraction(rgb_image_op)
        return tf.cast(rgb_image_op, tf.float32)

    def train(self, dataset=FLAGS.grasp_dataset, batch_size=1, epochs=FLAGS.epochs,
              load_weights=FLAGS.save_weights,
              save_weights=FLAGS.load_weights,
              make_model_fn=grasp_model.grasp_model,
              imagenet_mean_subtraction=FLAGS.imagenet_mean_subtraction,
              grasp_sequence_steps=None):
        """Train the grasping dataset

        This function depends on https://github.com/fchollet/keras/pull/6928

        # Arguments

            make_model_fn:
                A function of the form below which returns an initialized but not compiled model, which should expect
                input tensors.

                    make_model_fn(pregrasp_op_batch,
                                  grasp_step_op_batch,
                                  simplified_grasp_command_op_batch)

            grasp_sequence_steps: number of motion steps to train in the grasp sequence,
                this affects the memory consumption of the system when training, but if it fits into memory
                you almost certainly want the value to be None, which includes every image.
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        feature_op_dicts, features_complete_list, num_samples = data.get_simple_parallel_dataset_ops(batch_size=batch_size)
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()

        # TODO(ahundt) make "batches" also contain additional steps in the grasp attempt
        rgb_clear_view = data.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded',
            step='view_clear_scene'
        )

        rgb_move_to_grasp_steps = data.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded',
            step='move_to_grasp'
        )

        pose_op_params = data.get_time_ordered_features(
            features_complete_list,
            feature_type='params',
            step='move_to_grasp'
        )

        # print('features_complete_list: ', features_complete_list)
        grasp_success = data.get_time_ordered_features(
            features_complete_list,
            feature_type='grasp_success'
        )
        # print('grasp_success: ', grasp_success)

        # TODO(ahundt) Do we need to add some imagenet preprocessing here? YES when using imagenet pretrained weights
        # TODO(ahundt) THE NUMBER OF GRASP STEPS MAY VARY... CAN WE DEAL WITH THIS? ARE WE?

        # our training batch size will be batch_size * grasp_steps
        # because we will train all grasp step images w.r.t. final
        # grasp success result value
        pregrasp_op_batch = []
        grasp_step_op_batch = []
        # simplified_network_grasp_command_op
        simplified_grasp_command_op_batch = []
        grasp_success_op_batch = []
        # go through every element in the batch
        for fixed_feature_op_dict, sequence_feature_op_dict in feature_op_dicts:
            # print('fixed_feature_op_dict: ', fixed_feature_op_dict)
            # get the pregrasp image, and squeeze out the extra batch dimension from the tfrecord
            # TODO(ahundt) move squeeze steps into dataset api if possible
            pregrasp_image_rgb_op = fixed_feature_op_dict[rgb_clear_view[0]]
            pregrasp_image_rgb_op = self._rgb_preprocessing(pregrasp_image_rgb_op, imagenet_mean_subtraction=imagenet_mean_subtraction)

            grasp_success_op = tf.squeeze(fixed_feature_op_dict[grasp_success[0]])
            # each step in the grasp motion is also its own minibatch
            for i, (grasp_step_rgb_feature_name, pose_op_param) in enumerate(zip(rgb_move_to_grasp_steps, pose_op_params)):
                if grasp_sequence_steps is None or i < grasp_sequence_steps:
                    if int(grasp_step_rgb_feature_name.split('/')[1]) != int(pose_op_param.split('/')[1]):
                        raise ValueError('ERROR: the time step of the grasp step does not match the motion command params, '
                                         'make sure the lists are indexed correctly!')
                    pregrasp_op_batch.append(pregrasp_image_rgb_op)
                    grasp_step_rgb_feature_op = _rgb_preprocessing(fixed_feature_op_dict[grasp_step_rgb_feature_name])
                    grasp_step_op_batch.append(grasp_step_op)
                    simplified_grasp_command_op_batch.append(fixed_feature_op_dict[pose_op_param])
                    grasp_success_op_batch.append(grasp_success_op)

        pregrasp_op_batch = tf.parallel_stack(pregrasp_op_batch)
        grasp_step_op_batch = tf.parallel_stack(grasp_step_op_batch)
        simplified_grasp_command_op_batch = tf.parallel_stack(simplified_grasp_command_op_batch)
        grasp_success_op_batch = tf.parallel_stack(grasp_success_op_batch)

        pregrasp_op_batch = tf.concat(pregrasp_op_batch, 0)
        grasp_step_op_batch = tf.concat(grasp_step_op_batch, 0)
        simplified_grasp_command_op_batch = tf.concat(simplified_grasp_command_op_batch, 0)
        grasp_success_op_batch = tf.concat(grasp_success_op_batch, 0)
        # add one extra dimension so they match
        grasp_success_op_batch = tf.expand_dims(grasp_success_op_batch, -1)

        model = make_model_fn(
            pregrasp_op_batch,
            grasp_step_op_batch,
            simplified_grasp_command_op_batch
            )

        if(load_weights):
            if os.path.isfile(load_weights):
                model.load_weights(load_weights)
            else:
                print('Could not load weights {}, the file does not exist, starting fresh....'.format(load_weights))

        callbacks = []
        callbacks.append(ModelCheckpoint(save_weights + '.{epoch:03d}-{val_loss:.2f}.h5', save_best_only=True, verbose=1))

        # Nadam parameter choice:
        # https://github.com/tensorflow/tensorflow/pull/9175#issuecomment-295395355
        optimizer = keras.optimizers.Nadam(lr=0.004, beta_1=0.825, beta_2=0.99685)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[grasp_success_op_batch],
                      callbacks=callbacks)

        model.summary()

        # make sure we visit every image once
        steps_per_epoch = int(np.ceil(float(num_samples)/float(batch_size)))
        model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch)
        model.save_weights('grasp_model_weights.h5')


if __name__ == '__main__':

    with K.get_session() as sess:
        if FLAGS.grasp_model is 'grasp_model_single':
            model_fn = grasp_model.grasp_model
        elif FLAGS.grasp_model is 'grasp_model_segmentation':
            model_fn = grasp_model.grasp_model_segmentation
        else:
            raise ValueError('unknown model selected: {}'.format(FLAGS.grasp_model))

        gt = GraspTrain()
        gt.train(make_model_fn=model_fn, grasp_sequence_steps=1)
