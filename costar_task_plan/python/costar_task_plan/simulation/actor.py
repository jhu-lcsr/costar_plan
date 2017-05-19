
from costar_task_plan.abstract import AbstractAction
from costar_task_plan.abstract import AbstractActor
from costar_task_plan.abstract import AbstractState

class SimulationActorState(AbstractState):
    pass

class SimulationActorAction(AbstractAction):
    pass

class SimulationActor(AbstractActor):
    '''
    The simulation actor is an individual robot. Each actor is defined as:
     - a handle
     - a state
    '''
    def __init__(self, handle, *args, **kwargs):
        super(SimulationActor, self).__init__(*args, **kwargs)
        self.handle = handle
        
