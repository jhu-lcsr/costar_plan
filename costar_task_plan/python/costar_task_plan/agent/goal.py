
from abstract import AbstractAgent


class RandomGoalAgent(AbstractAgent):
    '''
    Reads goal information from the task and world. Will sample goals based
    on some guiding information provided by the goal and the task model.

    Unlike the others, this relies on having a working implementation of the
    "simulation.robot.RobotInterface" abstract class associated with the task.
    '''

    name = "random_goal"

    def __init__(self, max_iter=10000, *args, **kwargs):
        super(RandomGoalAgent, self).__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.task = None
        self.task_def = None

    def fit(self, env):
        '''
        Set the task and task model. We will sample from the task model at
        each step to move to our goals, with a random walk. We'll use any hints
        provided in the task model to help make sure we get somewhat useful
        information.
        '''
        self.task_def = env.client.task
        self.task = env.client.task.task
        for i in xrange(10000):
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
