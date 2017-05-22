
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
    This assumes the world is in the correct configuration, as represented
    by "state."
    '''
    def __call__(self, state, action):
        state.robot.act(action.cmd)

class SimulationRobotState(AbstractState):
    '''
    Includes full state necessary for this robot, including gripper, base, and 
    arm position.
    '''
    def __init__(self, robot,
            base_pos=(0,0,0),
            base_rot=(0,0,0,1),
            arm=[],
            gripper=0.,
            simulation_id=0):

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
    def __init__(self, cmd):
        self.cmd = cmd

class SimulationRobotActor(AbstractActor):
    def __init__(self, robot, *args, **kwargs):
        super(SimulationRobotActor, self).__init__(*args, **kwargs)
        self.robot = robot
