from abstract import AbstractTaskDefinition

import numpy as np
import os
import pybullet as pb
import rospkg


class ClutterTaskDefinition(AbstractTaskDefinition):

    '''
    Clutter task description in general
    '''

    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]
    sdf_dir = "sdf"
    model_file_name = "model.sdf"

    def __init__(self, *args, **kwargs):
        '''
        Your desription here
        '''
        super(ClutterTaskDefinition, self).__init__(*args, **kwargs)

    def _setup(self):
        '''
        Create random objects at random positions. Load random objects from the
        scene and create them in different places. In the future we may want to
        switch from using the list of "all" objects to a subset that we can
        actually pick up and manipulate.
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_objects')
        sdf_dir = os.path.join(path, self.sdf_dir)
        objs = [obj for obj in os.listdir(
            sdf_dir) if os.path.isdir(os.path.join(sdf_dir, obj))]

        randn = np.random.randint(1, len(objs))

        objs_to_add = np.random.choice(objs, randn)
        objs_to_add = [os.path.join(sdf_dir, obj, self.model_file_name)
                       for obj in objs_to_add]

        # load sdfs for all objects and initialize positions
        for obj in objs_to_add:
            try:
                pb.loadSDF(obj)
            except Exception, e:
                print e

    def _setupRobot(self, handle):
        '''
        Configure the robot so that it is ready to begin the task. Robot should
        be oriented so the gripper is near the cluttered area.
        '''
        for i, q in enumerate(self.joint_positions):
            pb.resetJointState(handle, i, q)
