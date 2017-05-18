from abstract import AbstractTaskDefinition

import numpy as np
import os
import pybullet as pb
import rospkg


class MugTaskDefinition(AbstractTaskDefinition):
    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]
    sdf_dir = "sdf"
    model_file_name = "model.sdf"
    model = "mug"

    spawn_pos_min = np.array([-0.4 ,-0.25, 0.05])
    spawn_pos_max = np.array([-0.65, 0.25, 0.055])
    spawn_pos_delta = spawn_pos_max - spawn_pos_min

    def __init__(self, *args, **kwargs):
        '''
        Your desription here
        '''
        super(MugTaskDefinition, self).__init__(*args, **kwargs)

    def _setup(self):
        '''
        Create the mug at a random position on the ground, handle facing
        roughly towards the robot. Robot's job is to grab and lift.
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_objects')
        sdf_dir = os.path.join(path, self.sdf_dir)
        obj_to_add = os.path.join(sdf_dir, self.model, self.model_file_name)

        identity_orientation = pb.getQuaternionFromEuler([0,0,0])
        try:
            obj_id_list = pb.loadSDF(obj_to_add)
            for obj_id in obj_id_list:
                random_position = np.random.rand(3)*self.spawn_pos_delta + self.spawn_pos_min
                pb.resetBasePositionAndOrientation(obj_id, random_position, identity_orientation)
        except Exception, e:
            print e

    def _setupRobot(self, handle):
        '''
        Configure the robot so that it is ready to begin the task. Robot should
        be oriented so the gripper is near the cluttered area with the mug.
        '''
        for i, q in enumerate(self.joint_positions):
            pb.resetJointState(handle, i, q)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

