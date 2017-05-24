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


class VREPGraspSimulation(object):

    def __init__(self):
        """Start the connection to the remote V-REP simulation
        """
        print 'Program started'
        # just in case, close all opened connections
        v.simxFinish(-1)
        # Connect to V-REP
        client_id = v.simxStart(FLAGS.vrepConnectionAddress,
                                FLAGS.vrepConnectionPort,
                                FLAGS.vrepWaitUntilConnected,
                                FLAGS.vrepDoNotReconnectOnceDisconnected,
                                FLAGS.vrepTimeOutInMs,
                                FLAGS.vrepCommThreadCycleInMs)
        if client_id != -1:
            print 'Connected to remote API server'
        return

    def visualize(self, tf_session, dataset=None):
        """Visualize one dataset in V-REP
        """
        grasp_dataset = GraspDataset(dataset)
        tf_glob = grasp_dataset.get_tfrecord_path_glob_pattern()
        features_complete_list = grasp_dataset.get_features()
        record_input = data_flow_ops.RecordInput(tf_glob)
        records_op = record_input.get_yield_op()
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()
        features_op_dict = grasp_dataset.parse_grasp_attempt_protobuf(features_complete_list, records_op)
        base_to_endeffector_transforms = grasp_dataset.get_time_ordered_features(
            features_complete_list,
            feature_type='transforms/base_T_endeffector/vec_quat_7')
        camera_to_base_transform = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics = 'camera/intrinsics/matrix33'
        tf_session.run(tf.global_variables_initializer())

        # TODO(ahundt) put a loop here
        output_features_dict = tf_session.run(features_op_dict)
        gripper_positions = [output_features_dict[transform_name] for transform_name in camera_to_base_transform]
        print gripper_positions
        print output_features_dict[camera_to_base_transform]
        print output_features_dict[camera_intrinsics]

    def __del__(self):
        v.simxFinish(-1)


if __name__ == '__main__':

    with tf.Session() as sess:
        sim = VREPGraspSimulation()
        sim.visualize(sess)

