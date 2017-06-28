
from costar_task_plan.abstract import *

import pybullet as pb
import PyKDL as kdl

class SimulationWorld(AbstractWorld):

    def __init__(self, dt = 0.0001, num_steps=1, save_hook=False, task_name="", cameras=[], *args, **kwargs):
        super(SimulationWorld, self).__init__(NullReward(), *args, **kwargs)
        self.num_steps = num_steps
        self.save_hook = save_hook
        self.task_name = task_name
        self.cameras = cameras

        # stores object handles and names
        self.class_by_object = {}
        self.object_by_class = {}
        self.id_by_object = {}

    def addObject(self, obj_name, obj_class, handle, state):
        '''
        Wraps add actor function for objects. Make sure they have the right
        policy and are added so we can easily look them up later on.
        '''
        
        obj_id = self.addActor(SimulationObjectActor(
            name=obj_name,
            handle=handle,
            dynamics=SimulationDynamics(self),
            policy=NullPolicy(),
            state=state))
        self.class_by_object[obj_name] = obj_class
        self.id_by_object[obj_name] = obj_id
        if obj_class not in self.object_by_class:
            self.object_by_class[obj_class] = [obj_name]
        else:
            self.object_by_class[obj_class].append(obj_name)

        return obj_id

    def getObjectId(self, obj_name):
      return self.id_by_object[obj_name]

    def getObjects(self):
        '''
        Return information about specific objects in the world. This should tell us
        for some semantic identifier which entities in the world correspond to that.
        As an example:
            {
                "goal": ["goal1", "goal2"]
            }
        Would be a reasonable response, saying that there are two goals called
        goal1 and goal2.
        '''
        return self.object_by_class

    def getObject(self, name):
        '''
        Look up the particular actor we are interested in for this world.

        Params:
        -------
        name: name of the actor (string)
        '''
        idx = self.id_by_object[name]
        return self.actors[idx]

    def hook(self):
        '''
        Step the simulation forward after all actors have given their comments
        to associated simulated robots. Then update all actors' states.
        '''

        # Loop through the given number of steps
        for i in xrange(self.num_steps):
            pb.stepSimulation()

        # Update the states of all actors.
        for actor in self.actors:
            actor.state = actor.getState()

    
    def zeroAction(self, actor):
        return SimulationRobotAction()

class SimulationDynamics(AbstractDynamics):
    '''
    Send robot's command over to the actor in the current simulation.
    This assumes the world is in the correct configuration, as represented
    by "state."
    '''
    def __call__(self, state, action, dt):
        if action.arm_cmd is not None :
            state.robot.arm(action.arm_cmd)
        if action.gripper_cmd is not None:
            state.robot.gripper(action.gripper_cmd)

class SimulationObjectState(AbstractState):
    '''
    Represents state and position of an arbitrary rigid object, and any
    associated predicates.
    '''
    def __init__(self, handle,
            base_pos=(0,0,0),
            base_rot=(0,0,0,1)):
        self.predicates = []
        self.base_pos = base_pos
        self.base_rot = base_rot
        p = kdl.Vector(*base_pos)
        R = kdl.Rotation.Quaternion(*base_rot)
        self.T = kdl.Frame(R,p)

class SimulationObjectActor(AbstractActor):
    '''
    Not currently any different from the default actor.
    '''

    def __init__(self, name, handle, *args, **kwargs):
        super(SimulationObjectActor, self).__init__(*args, **kwargs)
        self.name = name
        self.handle = handle
        self.getState = lambda: GetObjectState(self.handle)

class SimulationRobotState(AbstractState):
    '''
    Includes full state necessary for this robot, including gripper, base, and 
    arm position.
    '''
    def __init__(self, robot,
            base_pos=(0,0,0),
            base_rot=(0,0,0,1),
            arm=[],
            gripper=0.):

        self.predicates = []
        self.arm = arm
        self.gripper = 0.
        self.base_pos = base_pos
        self.base_rot = base_rot
        self.robot = robot

class SimulationRobotAction(AbstractAction):
    '''
    Includes the command that gets sent to robot.act()
    '''
    def __init__(self, arm_cmd=None, gripper_cmd=None):
        self.arm_cmd = arm_cmd
        self.gripper_cmd = gripper_cmd

class SimulationRobotActor(AbstractActor):
    def __init__(self, robot, *args, **kwargs):
        super(SimulationRobotActor, self).__init__(*args, **kwargs)
        self.robot = robot
        self.getState = self.robot.getState

class NullPolicy(AbstractPolicy):
  def evaluate(self, world, state, actor=None):
    return SimulationRobotAction()

# =============================================================================
# Helper Fucntions
def GetObjectState(handle):
    '''
    Look up the handle and get its base position and eventually other
    information, if we find that necessary.
    '''
    pos, rot = pb.getBasePositionAndOrientation(handle)
    return SimulationObjectState(handle,
                                 base_pos=pos,
                                 base_rot=rot)
