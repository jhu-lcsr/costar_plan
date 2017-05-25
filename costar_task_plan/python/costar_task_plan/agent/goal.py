
from abstract import AbstractAgent


class RandomGoalAgent(AbstractAgent):
    '''
    Reads goal information from the task and world. Will sample goals based
    on some guiding information provided by the goal and the task model.
    '''

    name = "random_goal"

    def __init__(self, iter=10000, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):
        for i in xrange(10000):
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
