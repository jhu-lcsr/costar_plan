"""Write features to a tensorflow tfrecord.

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""

import tensorflow as tf
import numpy as np


class TFRecordConverter(object):
    """Used to convert features for a machine learning dataset to the tfrecord format

    Usage:

      tfrecord_lambda_dict = None

      data1 = np.array([0,0])
      label1 = np.array(0)
      example1 = {'my_data': data1, 'my_label':label1}


      converter = TFRecordConverter(example1)



    """
    def __init__(self, data_file='data.tfrecord', initial_example=None):
        # maintain lambdas for lifetime of writing object
        self.tf_writer = tf.python_io.TFRecordWriter(data_file)
        self.feature_name_to_lambda = {}
        self.feature_to_dtype_converters = None
        if initial_example is not None:
            self.feature_to_dtype_converters = self.prepare_to_write(initial_example)

    def __del__(self):
        self.tf_writer.close()

    @staticmethod
    def _value_error(value):
        raise ValueError('Attempting to write unsupported data type {}, '
                         'add support for the new feature type '
                         'or update your code to write the data in a '
                         'supported format such as'
                         ' numpy arrays.'.format(str(type(value))))

    @staticmethod
    def feature_dtype_converter(value):
        """Create lambda functions matched to the appropriate
           tf.train.Feature class for the dtype of ndarray or integer value.

           The purpose of this function is for when you are writing the same tfrecord
           features over and over the expensive type checks do not need to be performed
           over and over because you can save these lambda functions that will generate
           the exact type you need for that feature.
           reference: https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f

           It is important to note that the numpy arrays you write must be 1 dimensional,
           so first write a shape array, then write the result of `array.flatten()`.
           This will ensure you write 2 1d arrays that together give you all the information
           you need to reconstruct your original nd array.
        """
        if type(value).__module__ == np.__name__:
            dtype_ = value.dtype
            if dtype_ == np.float64 or dtype_ == np.float32:
                return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
            elif dtype_ == np.int64 or dtype_ == np.int32:
                return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
            elif dtype_ == np.uint8 or dtype_ == np.int8:
                return lambda array: tf.train.Feature(bytes_list=tf.train.BytesList(value=array))
            else:
                TFRecordConverter._value_error(value)
        elif isinstance(value, int):
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=[array]))
        elif isinstance(value, float):
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=[array]))
        else:
            TFRecordConverter._value_error(value)

    def prepare_to_write(self, dict):
        """
        Create a dictionary of lambda function dictionaries, so type checks
        don't have to be re-done every time data is saved. Sorry this is a bit tricky,
        but we have to write out both the array and its shape when writing ndarrrays,
        so there must be two entries added for each single numpy ndarray.

        Here is a reference for updating dicts that might help with what's going on:
        https://stackoverflow.com/a/45043651/99379

        # Returns

            A dict containing the entries like following:
            {
                key: lambda_function(key, value)
            }


            Later if you have a dict with keys that are feature strings and values that are ndarrays,
            you can call dict[key](value). This will initialize the tf.train.Feature object you need
            to write out to a tfrecords file. See _tfrecord_write_example for a demonstration of usage.
        """
        features = {}
        for key, value in dict:
            if type(value).__module__ == np.__name__:
                # we need to write out both the shape of the array and then a flat version of it.
                # reference for calling the lambdas then saving them internally
                # https://stackoverflow.com/a/10452819/99379
                self.feature_name_to_lambda[key] = TFRecordConverter.feature_dtype_converter(value)
                self.feature_name_to_lambda[key + '_shape'] = TFRecordConverter.feature_dtype_converter(value)

                # create the function that will create the ndarray tfrecord features
                features[key] = lambda fkey, array: {
                    fkey + '_shape': self.feature_name_to_lambda[fkey + '_shape'](
                        np.array(array.shape, dtype=np.int64)),
                    fkey: self.feature_name_to_lambda[fkey](array.flatten())
                }
            elif isinstance(value, (int, float)):
                # reference for calling the lambdas then saving them internally
                # https://stackoverflow.com/a/10452819/99379
                self.feature_name_to_lambda[key] = TFRecordConverter.feature_dtype_converter(value)
                features[key] = lambda fkey, i: {fkey: self.feature_name_to_lambda[fkey](i)}
            else:
                TFRecordConverter._value_error(value)

        self.feature_to_dtype_converters = features

    def ready_to_write(self):
        """Returns if the converter is ready to write examples
        """
        if self.feature_to_dtype_converters is not None:
            return True
        else:
            return False

    def write_example(self, data_to_write):
        """Write an example to a file

        # Arguments

           data_to_write: A dictionary from feature strings to feature values.
               Values can be numpy arrays, floats, or integers.

        See also: feature_dict_dtype_converter
        """
        tf_train_features_dict = {}
        for key, value in data_to_write:
            # get the function that generates the feature,
            # call it, and update the feature dictionary
            # with the newly created feature object
            #
            # This is a bit complicated because with ndarrays
            # we want to create both an entry with 'feature_name'
            # and 'feature_name_shape', so the data can be easily
            # extracted later.
            tf_train_features_dict.update(self.feature_to_dtype_converters[key](key, value))

        features = tf.train.Features(feature=tf_train_features_dict)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        self.tf_writer.write(serialized)

    def close(self):
        """Close the underlying tfrecord writer self.tf_writer
        """
        self.tf_writer.close()
