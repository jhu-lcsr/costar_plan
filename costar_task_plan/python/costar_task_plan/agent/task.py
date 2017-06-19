from abstract import AbstractAgent


class TaskAgent(AbstractAgent):
    '''
    This agent uses a task model to generate a random sequence of actions. It
    requires a compiled task model (associated with the environment) for
    execution.

    An "instantiated task model" outlines a set of short-running policies that
    will take the robot towards its goals. This will use the "sample" function
    associated with each stage in the task plan to generate stochastic
    executions.
    '''

    name = "random"

    def __init__(self, iter=10000, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):

        task = env.taskModel()
        if not task.compiled:
            raise RuntimeError('environment must have associated compiled task model!')

        while True:
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
