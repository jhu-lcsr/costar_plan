
from costar_task_plan.abstract import *

class SimulationWorld(AbstractWorld):

    def __init__(self, client, dt = 0.02, num_steps=5, *args, **kwargs):
        super(SimulationWorld, self).__init__(client.getReward(), *args, **kwargs)
        self.client = client
        

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

