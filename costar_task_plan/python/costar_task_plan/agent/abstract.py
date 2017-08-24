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

import numpy as np
import os
import signal

class AbstractAgent(object):
    '''
    Default agent. Wraps a large number of different methods for learning a
    neural net model for robot actions.

    TO IMPLEMENT AN AGENT:
    - you mostly are just implementing _fit(). It must be able to handle the
      _break flag which will be caught by the higher level.
    '''

    name = None
    
    def __init__(self,
            env=None,
            verbose=False,
            save=False,
            load=False,
            directory='.',
            window_length=10,
            data_file='data.npz',
            *args, **kwargs):
        '''
        Sets up the general Agent.

        Params:
        ---------
        env: environment gym to run
        verbose: print out a ton of warnings and other information.
        save: save data collected to the disk somewhere.
        load: load data from the disk.
        '''

        self.env = env
        self._break = False
        self.verbose = verbose
        self.save = save
        self.load = load
        self.data = {}
        self.current_example = {}
        self.last_example = None

        self.datafile_name = data_file
        self.datafile = os.path.join(directory, data_file)
        if self.load:
            if os.path.isfile(self.datafile):
                self.data.update(np.load(self.datafile))
            elif self.load:
                raise RuntimeError('Could not load data from %s!' % \
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
            print "---- saving to %s ----"%self.datafile_name
            np.savez_compressed(self.datafile, **self.data)

    def _fit(self, num_iter):
        raise NotImplementedError('_fit() should run algorithm on' + \
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
            data = world.vectorize(control, features, reward, done, example,
                    action_label)
            self._updateCurrentExample(data)
            if done:
                self._finishCurrentExample(world)

    def _updateCurrentExample(self, data):
        for key,value in data:
            if not key in self.current_example:
                self.current_example[key] = [value]
            else:
                if isinstance(value, np.ndarray):
                    assert value.shape == self.current_example[key][0].shape
                if not type(self.current_example[key][0]) == type(value):
                    print key, type(self.current_example[key][0]), type(value)
                    raise RuntimeError('Types do not match when' + \
                                       ' constructing data set.')
                self.current_example[key].append(value)

    def _finishCurrentExample(self, world):
        '''
        Preprocess this particular example:
        - split it up into different time windows of various sizes
        - compute task result
        - compute transition points
        - compute option-level (mid-level) labels
        '''

        # Split into chunks and preprocess the data
        # This requires setting up window_length, etc

        next_list = world.features.description + ["reward", "label"]
        prev_list = ["label"]
        final_list = world.features.description
        length = len(self.current_example['example'])

        # ============================================
        # Loop over all entries. For important items, take the previous frame
        # and the next frame -- and possibly even the final frame.
        for i in xrange(length):
            i0 = max(i-1,0)
            i1 = min(i+1,length-1)
            ifinal = length-1

            # ==========================================
            # Finally, add the example to the dataset
            for key, values in self.current_example.items():
                if not key in self.data:
                    self.data[key] = []
                    if key in next_list:
                        self.data["next_%s"%key] = []
                    if key in prev_list:
                        self.data["prev_%s"%key] = []
                    if key in final_list:
                        self.data["final_%s"%key] = []
                else:
                    # Check data consistency
                    if len(self.data[key]) > 0:
                        if isinstance(values[0], np.ndarray):
                            assert values[0].shape == self.data[key][0].shape
                        if not type(self.data[key][0]) == type(values[0]):
                            print key, type(self.data[key][0]), type(values[0])
                            raise RuntimeError('Types do not match when' + \
                                               ' constructing data set.')

                    # Append list of features to the whole dataset
                    self.data[key].append(values[i])
                    if key in prev_list:
                        self.data["prev_%s"%key].append(values[i0])
                    if key in next_list:
                        self.data["next_%s"%key].append(values[i1])
                    if key in final_list:
                        self.data["final_%s"%key].append(values[ifinal])

        self.current_example = {}

