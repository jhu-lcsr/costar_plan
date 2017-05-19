
from costar_task_plan.abstract import *

class SimulationWorld(AbstractWorld):

    def __init__(self, client, reward, dt = 0.02, num_steps=5, *args, **kwargs):
        super(SimulationWorld, self).__init__(reward, *args, **kwargs)
        self.client = client

    def hook(self):
        for i in xrange(self.num_steps):
            pb.stepSimulation()
