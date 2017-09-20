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
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import signal

from costar_models.datasets.tfrecord import TFRecordConverter
from costar_models.datasets.npz import NpzDataset

class AbstractAgent(object):
    '''
    The AGENT handles data I/O, creation, and testing. Basically it is the
    interface from learning to the simulation.

    This class defines the basic, shared agent functionality. It wraps a large
    number of different methods for learning a neural net model for robot
    actions.

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
            window_length=10,
            trajectory_length=10,
            data_file='data.npz',
            data_type=None,
            success_only=False, # save all examples
            seed=0, # set a default seed
            collect_trajectories=False,
            collection_mode="goal",
            random_downsample=False,
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
        success_only: when loading data, only load successful examples.
                      Primarily intended for behavioral cloning.
        data_file: parsed to learn how to save data. Include either npz or
                   tfrecord after the period.
        directory: where to place saved data files
        seed: specify random seed for testing/validation/data collection.
        window_length: (not currently implemented) length of history to save
                       for each data point.
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
        self.current_example = {}
        self.last_example = None
        self.tfrecord_lambda_dict = None
        self.data_type = data_type
        self.seed = seed
        self.success_only = success_only
        self.random_downsample = random_downsample
        self.collect_trajectories = collect_trajectories
        self.collection_mode = collection_mode
        self.trajectory_length = trajectory_length
        if self.collection_mode == "goal" and self.collect_trajectories:
            raise RuntimeError("trajectories over future goals currently " + \
                               "not supported")

        if self.data_type == self.NUMPY_ZIP:
            root = ""
            for tok in data_file.split('.')[:-1]:
                root += tok
            self.npz_writer = NpzDataset(root)
        else:
            self.tf_writer = TFRecordConverter(data_file)

        self.datafile_name = data_file
        self.datafile = os.path.join(directory, data_file)

        if self.load:
            # =====================================================================
            # This is necessary for reading data in to the models.
            self.data = {}
            if self.data_type == self.NUMPY_ZIP:
                self.data = self.npz_writer.load(success_only=self.success_only)
            elif self.load:
                raise RuntimeError('Could not load data from %s!' %
                                   self.datafile)

    def _catch_sigint(self, *args, **kwargs):
      if self.verbose:
        print("Caught sigint!")
      self._break = True


    def fit(self, num_iter=1000):
        '''
        Basic "fit" function used by custom Agents. Override this if you do not
        want the saving, loading, signal-catching behavior we construct here.
        This function will run a number of experiments in different
        environments, updating a data store as it goes.

        Various agents provide support for reinforcement learning, supervised
        learning from demonstration, and others.

        Params:
        ------
        num_iter: optional param, number of experiments to run. Will be sent to
                  the specific agent being used.
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
            if self.data_type == self.TFRECORD:
                self.tf_writer.close()

    def _fit(self, num_iter):
        raise NotImplementedError('_fit() should run algorithm on'
                                  ' the environment')

    def _addToDataset(self, world, control, features, reward, done, example,
                      action_label, max_label=-1, seed=None,):
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
            data = world.vectorize(control, features, reward, done, example,
                    action_label)
            self._updateCurrentExample(data)
            if done:
                self._finishCurrentExample(world, example, reward, max_label,
                        seed)

    def _updateCurrentExample(self, data):
        '''
        Add to the current trial, so we can compute things over the whole
        experiment.

        Parameters:
        ----------
        data: vectorized list of saveable information: control, features,
              reward, done,  example, and (int) action label.
        '''
        for key,value in data:
            if not key in self.current_example:
                self.current_example[key] = [value]
            else:
                if isinstance(value, np.ndarray):
                    assert value.shape == self.current_example[key][0].shape
                if not type(self.current_example[key][0]) == type(value):
                    print(key, type(self.current_example[key][0]), type(value))
                    raise RuntimeError('Types do not match when' + \
                                       ' constructing data set.')
                self.current_example[key].append(value)

    def _finishCurrentExample(self, world, example, reward, max_label,
            seed=None):
        '''
        Preprocess this particular example:
        - split it up into different time windows of various sizes
        - compute task result
        - compute transition points
        - compute option-level (mid-level) labels
        '''
        print("Finishing example",example,seed)

        # ============================================================
        # Split into chunks and preprocess the data.
        # This may require setting up window_length, etc.
        # NOTE: removing some unnecessary features that we really dont need to
        # save. This ued to add world.features.description
        if not self.collect_trajectories:
            if self.collection_mode == "next":
                next_list = ["reward", "label"] + world.features.description
                goal_list = []
            elif self.collection_mode == "goal":
                goal_list = ["reward", "label"] + world.features.description
                next_list = []
        else:
            next_list = []
            goal_list = []

        # -- NOTE: you can add other features here in the future, but for now
        # we do not need these. Label gets some unique handling.
        prev_list  = []
        first_list = []
        length = len(self.current_example['example'])

        # Create an empty dict to hold all the data from this last trial.
        data = {}
        data["prev_label"] = []

        # Compute the points where the data changes from one label to the next
        # and save these points as "goals".
        switches = []
        count = 1
        label = self.current_example['label']
        for i in xrange(length):
            if i+1 == length:
                switches += [i] * count
                count = 1
            elif not label[i+1] == label[i]:
                switches += [i+1] * count
                count = 1
            else:
                count += 1

        assert(len(switches) == len(self.current_example['example']))

        # ============================================
        # Set up total reward
        total_reward = np.sum(self.current_example["reward"])
        data["value"] = [total_reward] * len(self.current_example["example"])

        if self.collect_trajectories:
            feature_shapes = {}
            for f in world.features.description:
                shape = self.current_example[f][0].shape
                feature_shapes[f] = (self.trajectory_length,) + shape

        # ============================================
        # Loop over all entries. For important items, take the previous frame
        # and the next frame -- and possibly even the final frame.
        prev_label = max_label
        for i in range(length):
            i0 = max(i-1,0)
            i1 = min(i+1,length-1)
            ifirst = 0

            if self.collect_trajectories:
                # collect a trajectory from this point going forward, out to
                # whatever length trajectories are (determined by command line
                # options)
                if i + 10 >= length:
                    break

                # Take the next N examples and save them as a single entry.
                # This is how we set up prediction for a sequence of images to
                # come.
                features = {}
                for f, shape in feature_shapes.items():
                    features[f] = np.zeros(shape)
                for j in range(self.trajectory_length):
                    features[f][j] = self.current_example[f][i+j]

            # We will always include frames where the label changed. We may or
            # may not include frames where the 
            if (self.current_example["label"][i0] == self.current_example["label"][i1] 
                    and not i0 == 0 
                    and not i1 == length - 1 
                    and self.random_downsample 
                    and not np.random.randint(2) == 0):
                        continue

            # ==========================================
            # Finally, add the example to the dataset
            for key, values in self.current_example.items():
                if not key in data:
                    data[key] = []
                    if key in next_list:
                        data["next_%s"%key] = []
                    if key in prev_list:
                        data["prev_%s"%key] = []
                    if key in first_list:
                        data["first_%s"%key] = []
                    if key in goal_list:
                        data["goal_%s"%key] = []
                    if key in features:
                        data["traj_%s"%key] = []

                # Check data consistency
                if len(data[key]) > 0:
                    if isinstance(values[0], np.ndarray):
                        assert values[0].shape == data[key][0].shape
                    if not type(data[key][0]) == type(values[0]):
                        print(key, type(data[key][0]), type(values[0]))
                        raise RuntimeError('Types do not match when' + \
                                           ' constructing data set.')

                # Append list of features to the whole dataset
                data[key].append(values[i])
                if self.collect_trajectories and key in features:
                    data["traj_%s"%key].append(features[key])
                if key == "label":
                    data["prev_%s"%key].append(prev_label)
                    prev_label = values[i]
                if key in prev_list:
                    data["prev_%s"%key].append(values[i0])
                if key in next_list:
                    data["next_%s"%key].append(values[i1])
                if key in first_list:
                    data["first_%s"%key].append(values[ifirst])
                if key in goal_list:
                    data["goal_%s"%key].append(values[switches[i]])

        # ===================================================================
        # Print out the seed associated with this example for reproduction, and
        # use it as part of the filename. If the seed is not provided, we will
        # set to the current example index.
        if seed is None:
            seed = example

        if not (self.success_only and reward <= 0.):
            # ================================================
            # Handle TF Records. We save here instead of at the end.
            if self.data_type == self.TFRECORD:

                # Write all entries in data set to the TF record.
                length = len(data.values()[0])
                for i in xrange(length):
                    sample = []
                    for key, values in data.items():
                        sample.append((key, values[i]))

                    # TF writer prepare a sample
                    if self.tf_writer.ready_to_write() is False:
                        self.tf_writer.prepare_to_write(sample)
                    self.tf_writer.write_example(sample)
            else:
                self.npz_writer.write(data, seed, reward)
        else:
            print("-- skipping bad example %d"%seed)
    
        # ================================================
        # Reset the current example.
        del self.current_example
        self.current_example = {}


