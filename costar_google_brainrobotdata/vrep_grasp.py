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
        self.client_id = v.simxStart(FLAGS.vrepConnectionAddress,
                                     FLAGS.vrepConnectionPort,
                                     FLAGS.vrepWaitUntilConnected,
                                     FLAGS.vrepDoNotReconnectOnceDisconnected,
                                     FLAGS.vrepTimeOutInMs,
                                     FLAGS.vrepCommThreadCycleInMs)
        if self.client_id != -1:
            print 'Connected to remote API server'
        return

    def visualize(self, tf_session, dataset=None, batch_size=1):
        """Visualize one dataset in V-REP
        """
        grasp_dataset = GraspDataset(dataset)
        feature_op_dicts, features_complete_list = grasp_dataset.get_simple_parallel_dataset_ops()
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()

        tf_session.run(tf.global_variables_initializer())
        output_features_dict = tf_session.run(feature_op_dicts)
        for features_dict_np, sequence_dict_np in output_features_dict:
            # TODO(ahundt) actually put transforms into V-REP or pybullet
            base_to_endeffector_transforms = grasp_dataset.get_time_ordered_features(
                features_complete_list,
                feature_type='transforms/base_T_endeffector/vec_quat_7')
            camera_to_base_transform = 'camera/transforms/camera_T_base/matrix44'
            camera_intrinsics = 'camera/intrinsics/matrix33'

            # TODO(ahundt) put a loop here
            # gripper_positions = [features_dict_np[transform_name] for transform_name in base_to_endeffector_transforms]
            for i, transform_name in zip(range(len(base_to_endeffector_transforms)), base_to_endeffector_transforms):
                # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
                empty_buffer = bytearray()
                gripper_pose = features_dict_np[transform_name]
                # 3 cartesian and 4 quaternion elements
                transform_display_name = str(i) + '_' + transform_name.replace('/', '-').split('-')[0]
                print transform_name, transform_display_name, gripper_pose
                res, retInts, retFloats, retStrings, retBuffer = v.simxCallScriptFunction(
                    self.client_id,
                    'remoteApiCommandServer',
                    v.sim_scripttype_childscript,
                    'createDummy_function',
                    [],
                    gripper_pose,
                    [transform_display_name],
                    empty_buffer,
                    v.simx_opmode_blocking)
                if res == v.simx_return_ok:
                    print ('Dummy handle: ', retInts[0])  # display the reply from V-REP (in this case, the handle of the created dummy)
                else:
                    print ('Remote function call failed')
            # print gripper_positions
            print features_dict_np[camera_to_base_transform]
            print features_dict_np[camera_intrinsics]
            # returnCode, dummyHandle = v.simxCreateDummy(self.client_id, 0.1, colors=None, operationMode=simx_opmode_blocking)


    def __del__(self):
        v.simxFinish(-1)


if __name__ == '__main__':

    with tf.Session() as sess:
        sim = VREPGraspSimulation()
        sim.visualize(sess)

