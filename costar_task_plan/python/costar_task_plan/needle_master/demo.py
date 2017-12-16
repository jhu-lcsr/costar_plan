# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:31:19 2015

@author: Chris
"""

import numpy as np
from nm_file import *

'''
Stores data for a single performance of a task.
You can pull this data out as a Numpy array.
'''


class Demo:

    def __init__(self, env_height, env_width, filename=None):

        self.t = None
        self.s = None
        self.u = None

        self.env_height = env_height
        self.env_width = env_width

        if not filename is None:
            handle = file(filename, 'r')
            (env, time) = ParseDemoName(filename)
            self.Load(handle)
            self.env = env

    def Draw(self):
        import matplotlib.pyplot as plt
        plt.plot(self.s[:, 0], self.s[:, 1])

    '''
    Load demonstration from a file
    '''

    def Load(self, handle):

        t = []
        s = []
        u = []

        data = handle.readline()
        while not data is None and len(data) > 0:
            data = [float(x) for x in data.split(',')]

            t.append(data[0])
            s.append(data[1:4])
            u.append(data[4:])

            data = handle.readline()

        '''
        NOTE: the android game puts things in sort of a weird frame of reference.
        This puts everything into the one we might expect.
        '''
        self.t = np.array(t)  # .transpose()
        self.s = np.array(s)  # .transpose()
        self.u = np.array(u)  # .transpose()
        self.s[:,2] = -1*self.s[:,2] + np.pi
        self.u[:,0] = -1*self.u[:,0]


