from costar_task_plan.abstract import NullReward
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.camera import *

import pybullet as pb
import rospkg
import os

class AbstractTaskDefinition(object):

    '''
    Defines how we create the world, gives us the reward function, etc. This
    builds off of the costar task plan library: it will allow us to create a
    World object that represents a particular environment, based off of the
    basic BulletWorld.
    '''

    def __init__(self, robot, seed=None, option=None, save=False, *args, **kwargs):
        '''
        We do not create a world here, but we may need to cache things or read
        them off of the ROS parameter server as necessary.
        '''
        self.seed = seed
        self.option = option
        self.robot = robot
        self.world = None
        self.save_world = save

        # local storage for object info
        self._objs_by_type = {}
        self._type_and_name_by_obj = {}
        self._cameras = []

    def addCamera(self, camera):
        assert isinstance(camera, Camera)
        self._cameras.append(camera)

    def capture(self):
        imgs = []
        for camera in self._cameras:
            imgs.append((camera.name, camera.capture()))
        return imgs

    def addObject(self, typename, obj_id):
        '''
        Create an object and add it to the world. This will automatically track
        and update useful information based on the object's current position
        during each world update.
        '''
        if typename not in self._objs_by_type:
            self._objs_by_type[typename] = [obj_id]
            num = 0
        else:
            num = len(self._objs_by_type[typename])
            self._objs_by_type[typename].append(obj_id)

        objname = "%s%03d"%(typename,num)
        self._type_and_name_by_obj[obj_id] = (typename, objname)

    def getName(self):
        raise NotImplementedError('should return name describing this task')

    def setup(self):
        '''
        Create task by adding objects to the scene, including the robot.
        '''
        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_simulation')
        static_plane_path = os.path.join(path,'meshes','world','plane.urdf')
        pb.loadURDF(static_plane_path)

        self.task = self._makeTask()
        self.world = SimulationWorld(
                save_hook=self.save_world,
                task_name=self.getName())
        self._setup()
        handle = self.robot.load()
        pb.setGravity(0,0,-9.807)
        self._setupRobot(handle)

        state = self.robot.getState()
        self.world.addActor(SimulationRobotActor(
            robot=self.robot,
            dynamics=SimulationDynamics(self.world),
            policy=NullPolicy(),
            state=state))

        self._updateWorld()

        for handle, (obj_type, obj_name) in self._type_and_name_by_obj.items():
            # Create an object and add it to the World
            state = GetObjectState(handle)
            self.world.addObject(obj_name, obj_type, handle, state)

        self.task.compile(self.world)

    def reset(self):
        '''
        Reset the whole simulation into a working configuration.
        '''
        raise NotImplementedError('Task must override the reset() function!')

    def getReward(self):
        return NullReward()

    def _setup(self):
        '''
        Setup any world objects after the robot has been created.
        '''
        raise NotImplementedError('Must override the _setup() function!')

    def _setupRobot(self, handle):
        '''
        Do anything you need to do to the robot before it
        '''
        raise NotImplementedError('Must override the _setupRobot() function!')

    def _updateWorld(self):
        '''
        Do anything necessary to add other agents to the world.
        '''
        pass

    def _makeTask(self):
        '''
        Create the task model that we are attempting to learn policies for.
        '''
        raise NotImplementedError('Must override the _makeTask() function!')

    def cloneRobot(self):
        robot_type = type(self.robot)
        return robot_type()
