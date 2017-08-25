import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
import grasp_dataset
import grasp_model

from tensorflow.python.platform import flags


tf.flags.DEFINE_string('grasp_model', 'grasp_model_single',
                       """Choose the model definition to run, options are grasp_model and grasp_model_segmentation""")

FLAGS = flags.FLAGS


class GraspTrain(object):

    def _imagenet_preprocessing(self, tensor):
        """Do imagenet preprocessing, but make sure the network you are using needs it!

           zero centers by mean pixel.
        """
        # TODO(ahundt) do we need to divide by 255 to make it floats from 0 to 1? It seems no based on https://keras.io/applications/
        # TODO(ahundt) apply resolution to https://github.com/fchollet/keras/pull/7705 when linked PR is closed
        pixel_value_offset = tf.constant([103.939, 116.779, 123.68])
        return tf.subtract(tensor, pixel_value_offset)

    def train(self, dataset=FLAGS.grasp_dataset, steps_per_epoch=1000, batch_size=1, epochs=10, load_weights="", save_weights='grasp_model_weights.h5',
              imagenet_preprocessing=True,
              make_model_fn=grasp_model.grasp_model):

        """Visualize one dataset in V-REP
        """
        data = grasp_dataset.GraspDataset(dataset=dataset)
        # list of dictionaries the length of batch_size
        feature_op_dicts, features_complete_list = data.get_simple_parallel_dataset_ops(batch_size=batch_size)
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
        # print('pose_op_params: ', pose_op_params)

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
            pregrasp_op = tf.cast(tf.squeeze(fixed_feature_op_dict[rgb_clear_view[0]]), tf.float32)
            if imagenet_preprocessing:
                pregrasp_op = self._imagenet_preprocessing(pregrasp_op)
            grasp_success_op = tf.squeeze(fixed_feature_op_dict[grasp_success[0]])
            # each step in the grasp motion is also its own minibatch
            for grasp_step, pose_op_param in zip(rgb_move_to_grasp_steps, pose_op_params):
                pregrasp_op_batch.append(pregrasp_op)
                grasp_step_op = tf.cast(tf.squeeze(fixed_feature_op_dict[grasp_step]), tf.float32)
                if imagenet_preprocessing:
                    grasp_step_op = self._imagenet_preprocessing(grasp_step_op)
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
            model.load_weights(load_weights)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      target_tensors=[grasp_success_op_batch]
                      )

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
        gt.train(make_model_fn=model_fn)
