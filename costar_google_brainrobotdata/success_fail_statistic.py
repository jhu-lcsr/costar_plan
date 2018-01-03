import tensorflow as tf
import grasp_dataset


def count_success_failure_number(grasp, tf_session=tf.Session()):
    """ Counting number of success and failure in all attempts.
        Return: success_num, failure_num, success/failure ratio
    """
    batch_size = 1
    (feature_op_dicts, _, _, num_samples) = self.get_training_dictionaries(batch_size=batch_size)
    tf_session.run(tf.global_variables_initializer())

    success_num = 0
    failure_num = 0
    for attempt_num in range(num_samples):
        output_features_dicts = tf_session.run(feature_op_dicts)
        [(features_dict_np, _)] = output_features_dicts
        if int(features_dict_np['grasp_success']) == 1:
            success_num += 1
        else:
            failure_num += 1

    return success_num, failure_num, succes