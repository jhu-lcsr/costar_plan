from abstract import AbstractTaskDefinition

import numpy as np
import os
import pybullet as pb
import rospkg


class ClutterTaskDefinition(AbstractTaskDefinition):

    '''
    Clutter task description in general. This task should create a bunch of
    objects and bins to put them all in.
    '''

    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]
    sdf_dir = "sdf"
    model_file_name = "model.sdf"
    list_of_models_to_manipulate = ['c_clamp', 'drill_blue_small', 'driller_point_metal', 
        'driller_small', 'hammer', 'handspot', 'keyboard', 'mallet_ball_pein',
        'mallet_black_white', 'mallet_drilling', 'mallet_fiber',
        'mug', 'old_hammer', 'pepsi_can', 'sander']
    models = set(list_of_models_to_manipulate)
    spawn_pos_min = np.array([-0.4 ,-0.25, 0.1])
    spawn_pos_max = np.array([-0.65, 0.25, 0.3])
    spawn_pos_delta = spawn_pos_max - spawn_pos_min

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

        objs_name_to_add = np.random.choice(objs, randn)
        objs_to_add = [os.path.join(sdf_dir, obj, self.model_file_name)
                       for obj in objs_name_to_add]

        identity_orientation = pb.getQuaternionFromEuler([0,0,0])
        # load sdfs for all objects and initialize positions
        for obj_index, obj in enumerate(objs_to_add):
            if objs_name_to_add[obj_index] in self.models:
                try:
                    print 'Loading object: ', obj
                    obj_id_list = pb.loadSDF(obj)
                    for obj_id in obj_id_list:
                        random_position = np.random.rand(3)*self.spawn_pos_delta + self.spawn_pos_min
                        pb.resetBasePositionAndOrientation(obj_id, random_position, identity_orientation)
                except Exception, e:
                    print e

    def _setupRobot(self, handle):
        '''
        Configure the robot so that it is ready to begin the task. Robot should
        be oriented so the gripper is near the cluttered area.
        '''
        for i, q in enumerate(self.joint_positions):
            pb.resetJointState(handle, i, q)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, self.POSITION_CONTROL)
