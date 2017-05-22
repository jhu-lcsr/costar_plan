
from costar_task_plan.abstract import *

class SimulationWorld(AbstractWorld):

    def __init__(self, dt = 0.02, num_steps=1, *args, **kwargs):
        super(SimulationWorld, self).__init__(None, *args, **kwargs)

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
            pass
    
    def zeroAction(self, actor):
        pass

class SimulationDyamics(AbstractDynamics):
    '''
    Send robot's command over to the actor in the current simulation.
    '''
    def __call__(self, state, action):
        state.robot.act(action)

class SimulationRobotState(AbstractState):
    '''
    Includes full state necessary for this robot, including gripper, base, and 
    arm position.
    '''
    def __init__(self, robot, simulation_id=0):
        pass

class SimulationRobotAction(AbstractAction):
    '''
    Includes the command that gets sent to robot.act()
    '''
    def __init__(self, cmd):
        self.cmd = cmd

class SimulationRobotActor(AbstractActor):
    def __init__(self, robot, *args, **kwargs):
        self.robot = robot
