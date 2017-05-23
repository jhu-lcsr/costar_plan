"""Code for building the input for the prediction model."""

import os
import errno

import numpy as np

try:
    import vrep.vrep as v
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in PYTHONPATH folder relative to this file,')
    print ('or appropriately adjust the file "vrep.py. Also follow the"')
    print ('ReadMe.txt in the vrep remote API folder')
    print ('--------------------------------------------------------------')
    print ('')

import tensorflow as tf
import re

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from keras.utils import get_file

import moviepy.editor as mpy
from grasp_dataset import GraspDataset

tf.flags.DEFINE_string('vrepConnectionAddress', '127.0.0.1', 'The IP address of the running V-REP simulation.')
tf.flags.DEFINE_integer('vrepConnectionPort', 19999, 'ip port for connecting to V-REP')
tf.flags.DEFINE_boolean('vrepWaitUntilConnected', True, 'block startup call until vrep is connected')
tf.flags.DEFINE_boolean('vrepDoNotReconnectOnceDisconnected', True, '')
tf.flags.DEFINE_integer('vrepTimeOutInMs', 5000, 'Timeout in milliseconds upon which connection fails')
tf.flags.DEFINE_integer('vrepCommThreadCycleInMs', 5, 'time between communication cycles')

FLAGS = flags.FLAGS


class VREPSimulation:

    def __init__(self):
        return

    def start(self):
        """Start the connection to the remote V-REP simulation
        """
        print 'Program started'
        # just in case, close all opened connections
        v.simxFinish(-1)
        # Connect to V-REP
        clientID = v.simxStart(FLAGS.vrepConnectionAddress,
                               FLAGS.vrepConnectionPort,
                               FLAGS.vrepWaitUntilConnected,
                               FLAGS.vrepDoNotReconnectOnceDisconnected,
                               FLAGS.vrepTimeOutInMs,
                               FLAGS.vrepCommThreadCycleInMs)
        if clientID != -1:
            print 'Connected to remote API server'


if __name__ == '__main__':

    with tf.Session() as sess:
        gd = GraspDataset()
        gd.download(FLAGS.grasp_download)
        sim = VREPSimulation()
        sim.start()

        gd.create_gif(sess)

