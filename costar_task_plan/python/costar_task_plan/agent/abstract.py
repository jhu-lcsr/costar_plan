'''
By Chris Paxton
Copyright (c) 2017, The Johns Hopkins University
All rights reserved.

This license is for non-commercial use only, and applies to the following
people associated with schools, universities, and non-profit research institutions

Redistribution and use in source and binary forms by the aforementioned
people and institutions, with or without modification, are permitted
provided that the following conditions are met:

* Usage is non-commercial.

* Redistribution should be to the listed entities only.

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import tensorflow as tf
import numpy as np
import os
import signal


def _tfrecord_value_error(value):
            raise ValueError('Attempting to write unsupported data type {}, '
                             'add support for the new feature type '
                             'or update your code to write the data in a '
                             'supported format such as'
                             ' numpy arrays.'.format(str(type(value))))


def _tfrecord_dtype_feature(value):
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
            _tfrecord_value_error(value)
    elif isinstance(value, int):
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    elif isinstance(value, float):
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    else:
        _tfrecord_value_error(value)


def _tfrecord_dtype_lambda_dict(dict):
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
    class tfrecord_dtypes:
        # maintain lambdas for lifetime of
        # writing records, written as nonlocal
        # class definition for python 2.7 compatibility
        feature_name_to_lambda = {}
    for key, value in dict:
        if type(value).__module__ == np.__name__:
            # reference for calling the lambdas then saving them internally
            # https://stackoverflow.com/a/10452819/99379
            tfrecord_dtypes.feature_name_to_lambda[key] = _tfrecord_dtype_feature(value)
            value_shape = np.array(value.shape, dtype=np.int64)
            tfrecord_dtypes.feature_name_to_lambda[key + '_shape'] = _tfrecord_dtype_feature(value)
            key_shape_lambda = _tfrecord_dtype_feature(value_shape)
            features[key] = lambda fkey, array: {
                fkey + '_shape': tfrecord_dtypes.feature_name_to_lambda[fkey + '_shape'](
                    np.array(array.shape, dtype=np.int64)),
                fkey: tfrecord_dtypes.feature_name_to_lambda[fkey](array.flatten())
            }
        elif isinstance(value, (int, float)):
            # reference for calling the lambdas then saving them internally
            # https://stackoverflow.com/a/10452819/99379
            key_feature_lambda = _tfrecord_dtype_feature(value)
            features[key] = lambda fkey, i: {fkey: key_feature_lambda(i)}
        else:
            _tfrecord_value_error(value)

    return features


def _tfrecord_write_example(writer, dtype_lambda_dict, data_to_write):
    """Write an example to a file
    See also: _tfrecord_dtype_lambda_dict
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
        tf_train_features_dict.update(dtype_lambda_dict[key](key, value))

    features = tf.train.Features(feature=tf_train_features_dict)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)


class AbstractAgent(object):
    '''
    Default agent. Wraps a large number of different methods for learning a
    neural net model for robot actions.

    TO IMPLEMENT AN AGENT:
    - you mostly are just implementing _fit(). It must be able to handle the
      _break flag which will be caught by the higher level.
    '''

    name = None
    NULL = ''
    NUMPY_ZIP = 'npz'
    TFRECORD = 'tfrecord'

    def __init__(self,
            env=None,
            verbose=False,
            save=False,
            load=False,
            directory='.',
            data_file='data.npz',
            data_type=None,
            *args, **kwargs):
        '''
        Sets up the general Agent.

        Params:
        ---------
        env: environment gym to run
        verbose: print out a ton of warnings and other information.
        save: save data collected to the disk somewhere.
        load: load data from the disk.
        data_type: options are 'npz', 'tfrecord, None. The default None
            tries to detect the data type based on the data file extension,
            with .npz meaning the numpy zip format, and tfrecord meaning the
            tensorflow tfrecord format.
        '''
        if data_type is None:
            if '.npz' in data_file:
                data_type = self.NUMPY_ZIP
            elif '.tfrecord' in data_file:
                data_type = self.TFRECORD
            else:
                raise RuntimeError('Currently supported file extensions '
                                   'are .tfrecord and .npz, you entered '
                                   '{}'.format(data_file.split('.')[-1]))
        self.env = env
        self._break = False
        self.verbose = verbose
        self.save = save
        self.load = load
        self.last_example = None
        self.tfrecord_lambda_dict = None
        self.data_type = data_type

        if self.data_type == self.NUMPY_ZIP:
            self.data = {}
        else:
            self.tf_writer = tf.python_io.TFRecordWriter(data_file)

        self.datafile_name = data_file
        self.datafile = os.path.join(directory, data_file)
        if self.load:
            if os.path.isfile(self.datafile) and self.data_type == self.NUMPY_ZIP:
                self.data.update(np.load(self.datafile))
            elif self.load:
                raise RuntimeError('Could not load data from %s!' %
                                   self.datafile)

    def _catch_sigint(self, *args, **kwargs):
      if self.verbose:
        print "Caught sigint!"
      self._break = True



    def fit(self, num_iter=1000):
        '''
        Basic "fit" function used by custom Agents. Override this if you do not
        want the saving, loading, signal-catching behavior we construct here.

        Params:
        ------
        [none]
        '''
        self.last_example = None
        self.env.world.verbose = self.verbose
        self._break = False
        #_catch_sigint = lambda *args, **kwargs: self._catch_sigint(*args, **kwargs)
        #signal.signal(signal.SIGINT, _catch_sigint)
        try:
            self._fit(num_iter)
        except KeyboardInterrupt, e:
            pass

        if self.save:
            if self.data_type == self.NUMPY_ZIP:
                print("---- saving to %s ----" % self.datafile_name)
                np.savez_compressed(self.datafile, **self.data)
            if self.data_type == self.TFRECORD:
                self.tf_writer.close()

    def _fit(self, num_iter):
        raise NotImplementedError('_fit() should run algorithm on'
                                  ' the environment')

    def _addToDataset(self, world, control, features, reward, done, example,
                      action_label):
        '''
        Takes as input features, reward, action, and other information. Saves
        all of this to create a dataset. Any custom agents should call this
        function to update the dataset.

        Params:
        ----------
        world: the current world state
        control: the command send to the learning actor in the world.
        features: observations, information we saw before taking this action.
        reward: instantaneous reward.
        done: are we finished here?
        action_label: string data provided by the agent.
        '''

        # Save both the generic, non-parameterized action name and the action
        # name.
        world = self.env.world
        if self.save:
            # Features can be either a tuple or a numpy array. If they're a
            # tuple, we handle them one way...
            data = world.vectorize(control, features, reward, done, example, action_label)
            self._updateDatasetWithSample(data)

    def _updateDatasetWithSample(self, data):
        '''
        Helper function. Currently writes data to a big dictionary, which gets
        written out to a numpy archive.
        '''
        if self.data_type == self.TFRECORD:
            if self.tfrecord_lambda_dict is None:
                self.tfrecord_lambda_dict = _tfrecord_dtype_lambda_dict(data)

            _tfrecord_write_example(self.tf_writer, self.tfrecord_lambda_dict, data)

        elif self.data_type == self.NUMPY_ZIP:
            for key, value in data:
                    if key not in self.data:
                        self.data[key] = [value]
                    else:
                        if isinstance(value, np.ndarray):
                            assert value.shape == self.data[key][0].shape
                        if not type(self.data[key][0]) == type(value):
                            print(key, type(self.data[key][0]), type(value))
                            raise RuntimeError('Types do not match when' + \
                                               ' constructing data set.')
                        self.data[key].append(value)
        else:
            raise RuntimeError('file extension not recognized: %s'%self.data_type)

