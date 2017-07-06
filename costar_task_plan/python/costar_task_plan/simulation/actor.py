
from costar_task_plan.abstract import AbstractAction
from costar_task_plan.abstract import AbstractActor
from costar_task_plan.abstract import AbstractDynamics
from costar_task_plan.abstract import AbstractState

import pybullet as pb


class SimulationActorState(AbstractState):

    '''
    This represents the current position, velocity, abnd vase position of an
    actor in the world.
    '''

    def __init__(self, handle):
        self.handle = handle
        self.position = []
        self.velocity = []


class SimulationActorAction(AbstractAction):

    '''
    This represents the command that should be send to the actor next. It may
    be a position, or something else.
    '''

    def __init__(self):
        self.arm_cmd = []
        self.arm_mode = pb.POSITION_CONTROL


class SimulationActor(AbstractActor):

    '''
    The simulation actor is an individual robot. Each actor is defined as:
     - a handle
     - a state
    '''

    def __init__(self, handle, *args, **kwargs):
        super(SimulationActor, self).__init__(*args, **kwargs)
        self.handle = handle


class SimulationActorClientDynamics(AbstractDynamics):

    '''
    These dynamics send the appropriate command for the actor over to the
    server but do nothing else. They do not even update the state.
    '''

    def __init__(self, world, robot):
        '''
        Store the world, but also associated robot. This is what we will
        actually use to control the simulation.
        '''
        self.world = world
        self.robot = robot

    def apply(self, state, action):
        '''
        Send robot commands to the simulation
        '''
        self.robot.arm(action.arm_cmd, action.arm_mode)

        return None
