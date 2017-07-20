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
            verbose=False,
            save=False,
            load=False,
            directory='.',
            data_file='data.npz',
            *args, **kwargs):
        '''
        Sets up the general Agent.

        Params:
        ---------
        verbose: print out a ton of warnings and other information.
        save: save data collected to the disk somewhere.
        load: load data from the disk.
        '''

        self._break = False
        self.verbose = verbose
        self.save = save
        self.load = load
        self.data = {}

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
        [none]
        '''
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
        #generic_action_name = action_label.split('(')[0]
        world = self.env.world
        if self.save:
            # Features can be either a tuple or a numpy array. If they're a
            # tuple, we handle them one way...
            data = world.vectorize(control, features, reward, done, example,
                    action_label)

            for key, value in data:
                if not key in self.data:
                    self.data[key] = [value]
                else:
                    if isinstance(value, np.ndarray):
                        assert value.shape == self.data[key][0].shape
                    if not type(self.data[key][0]) == type(value):
                        print key, type(self.data[key][0]), type(value)
                        raise RuntimeError('Types do not match when' + \
                                           ' constructing data set.')
                    self.data[key].append(value)

class NoAgent(AbstractAgent):
    '''
    This does basically nothing. Use it to move on with your life and take the
    data set to train something else.
    '''
    def _fit(self, num_iter):
        pass
