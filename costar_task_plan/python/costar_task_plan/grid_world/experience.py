
import numpy as np
import random
import tensorflow as tf

'''
just a little structure to hold extra data
'''
class ExperienceInput(object):
    def __init__(self, niter, prev_fs, rs, actions, next_fs, terminal):
        self.niter = niter
        self.prev_fs = prev_fs
        self.rs = rs
        self.actions = actions
        self.next_fs = next_fs
        self.terminal = terminal

'''
This class stores an experience memory for deep RL.
'''
class Experience(object):

    def __init__(self, size, xdim, ydim, discount=0.9):
        self._prev_x = np.zeros((size, xdim))
        self._next_x = np.zeros((size, xdim))
        self._y = np.zeros((size, ydim))
        self._r = np.zeros((size, 1))
        self._terminal = np.zeros((size, 1), dtype=bool)
        self._idx = 0
        self._length = 0
        self._size = size
        self._discount = discount
        self.model_output = None
        self.model_input = None
        self.max_output = None

    def addInput(self, data):
        for i in xrange(data.niter):
            idx = (self._idx + i) % self._size
            self._prev_x[idx] = data.prev_fs[i]
            self._next_x[idx] = data.next_fs[i]
            self._r[idx]  = data.rs[i]
            self._terminal[idx] = data.terminal[i]
            self._y[idx]  = data.actions[i]

        self._idx += data.niter
        if self._length < self._size and self._idx > self._length:
            self._length = self._idx
            if self._length > self._size:
                self._length = self._size
        self._idx %= self._size

    '''
    this loads supervised training data
    we handle this a little differently from our other data
    '''
    def initFromArrays(self, x, y):
        length = x.shape[0]
        self._prev_x[:length,:] = x
        self._terminal[:length] = True
        self._y[:length,:] = y
        self._r[:length] = 1. # max reward for supervised learning
        self._idx = length
        self._length = length

    def setModel(self, model_output, model_input):
        self.model_output = model_output
        self.max_output = tf.reduce_max(model_output, reduction_indices=1)
        self.model_input = model_input

    '''
    withdraw a single minibatch
    also computes the rewards for training
    '''
    def sample(self, num):
        idx = random.sample(range(self._length), num)
        
        x = self._prev_x[idx]
        y = self._y[idx]
        r = self._r[idx]

        non_terminal_idx = np.squeeze(self._terminal[idx] == False)
        x1 = self._next_x[idx,:]
        x1nt = x1[non_terminal_idx]

        num_non_terminal = len(x1nt)
        if num_non_terminal > 0:
            best = self.max_output.eval(feed_dict={self.model_input: x1nt})
            r[non_terminal_idx] += (self._discount * best).reshape(num_non_terminal, 1)
            # print r[non_terminal_idx]
        

        # r[non_terminal_idx] = 

        # for i in idx
        #    if not self._terminal[i]:
        # add in estimated for next step
        #        r[i] = self._r[i] + 0

        return x, y, r
        

