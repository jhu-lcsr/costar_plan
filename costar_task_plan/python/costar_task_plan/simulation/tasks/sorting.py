from abstract import AbstractTaskDefinition
from default import DefaultTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *

import numpy as np
import os
import pybullet as pb
import rospkg


class SortingTaskDefinition(DefaultTaskDefinition):
    '''
    Define the simple sorting task.
    '''

    blue_urdf = "blue.urdf"
    red_urdf = "red.urdf"
    model = "ball"

    tray_dir = "tray"
    tray_urdf = "traybox.urdf"

    spawn_pos_min = np.array([-0.4 ,-0.25, 0.10])
    spawn_pos_max = np.array([-0.65, 0.25, 0.155])
    spawn_pos_delta = spawn_pos_max - spawn_pos_min

    tray_poses = [np.array([-0.5, 0., 0.0]),
                  np.array([0., +0.6, 0.0]),
                  np.array([-1.0, -0.6, 0.0])]

    def __init__(self, robot, red=3, blue=3, *args, **kwargs):
        '''
        Your desription here
        '''
        super(SortingTaskDefinition, self).__init__(robot, *args, **kwargs)
        self.num_red = red
        self.num_blue = blue

    def _makeTask(self):
        GraspOption = lambda goal: GoalDirectedMotionOption(
                self.world,
                goal,
                pose=((0.05,0,0),(0,0,0,1)))
        grasp_args = {
                "constructor": GraspOption,
                "args": ["red"],
                "remap": {"red": "goal"},
                }
        LiftOption = lambda: GeneralMotionOption(None)
        lift_args = {
                "constructor": LiftOption,
                "args": []
                }
        WaitOption = lambda: RelativeMotionOption(None)
        wait_args = {
                "constructor": GeneralMotionOption,
                "args": []
                }
        PlaceOption = lambda: GeneralMotionOption(None)
        place_args = {
                "constructor": PlaceOption,
                "args": []
                }
        close_gripper_args = {
                "constructor": PlaceOption,
                "args": []
                }
        open_gripper_args = {
                "constructor": PlaceOption,
                "args": []
                }

        # Create a task model
        task = Task()
        task.add("grasp", None, grasp_args)
        task.add("close_gripper", "grasp", close_gripper_args)
        task.add("lift", "close_gripper", grasp_args)
        task.add("place", "lift", grasp_args)
        task.add("open_gripper", "place", open_gripper_args)

        return task


    def _setup(self):
        '''
        Create the mug at a random position on the ground, handle facing
        roughly towards the robot. Robot's job is to grab and lift.
        '''


        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_objects')
        urdf_dir = os.path.join(path, self.urdf_dir)
        tray_filename = os.path.join(urdf_dir, self.tray_dir, self.tray_urdf)
        red_filename = os.path.join(urdf_dir, self.model, self.red_urdf)
        blue_filename = os.path.join(urdf_dir, self.model, self.blue_urdf)

        for position in self.tray_poses:
            obj_id = pb.loadURDF(tray_filename)
            pb.resetBasePositionAndOrientation(obj_id, position, (0,0,0,1))

        self._add_balls(self.num_red, red_filename, "red")
        self._add_balls(self.num_blue, blue_filename, "blue")

    def reset(self):
        for obj_id, position in zip(self.trays, self.tray_poses):
            pb.resetBasePositionAndOrientation(obj_id, position, (0,0,0,1))
        for obj_id in zip(self.balls):
            obj_id = pb.loadURDF(filename)
            random_position = np.random.rand(3)*self.spawn_pos_delta + self.spawn_pos_min
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)
        self.robot2 = self.cloneRobot()
        self.robot2.place([-1,0,0],[0,0,1,0],
                self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

    def _add_balls(self, num, filename, typename):
        '''
        Helper function to spawn a whole bunch of random balls.
        '''
        for i in xrange(num):
            obj_id = pb.loadURDF(filename)
            random_position = np.random.rand(3)*self.spawn_pos_delta + self.spawn_pos_min
            pb.resetBasePositionAndOrientation(obj_id, random_position, (0,0,0,1))
            self.addObject(typename, obj_id)

    def _setupRobot(self, handle):
        '''
        Configure the robot so that it is ready to begin the task. Robot should
        be oriented so the gripper is near the cluttered area with the mug.
        '''
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)
        self.robot2 = self.cloneRobot()
        self.robot2.load()
        self.robot2.place([-1,0,0],[0,0,1,0],
                self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

    def _updateWorld(self):
        '''
        Add the other robot, and actors for the different objects. These are
        mostly to add them to the update loops - so we can compute features that
        are relevant to whatever we actually want to do.
        '''
        state = self.robot2.getState()
        self.world.addActor(SimulationRobotActor(
            robot=self.robot2,
            dynamics=SimulationDynamics(self.world),
            policy=NullPolicy(),
            state=state))

    def getName(self):
        return "sorting"
