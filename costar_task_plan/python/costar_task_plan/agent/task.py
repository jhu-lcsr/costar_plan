from abstract import AbstractAgent
from costar_task_plan.mcts import ContinuousSamplerTaskPolicies
from costar_task_plan.mcts import RandomSearch
from costar_task_plan.mcts import Node


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

    def __init__(self, env, iter=10000, *args, **kwargs):
        super(TaskAgent, self).__init__(*args, **kwargs)
        self.iter = iter
        self.env = env

    def _addToDataset(self, action_name, world, control, features, reward, done):
        '''
        Takes as input features, reward, action, and other information. Saves
        all of this to create a dataset.
        '''

        # Save both the generic, non-parameterized action name and the action
        # name.
        generic_action_name = action_name.split('(')[0]

    def fit(self):
        '''
        This is a "fake" agent -- it does not fit any model in particular, it
        just generates some data. You almost certainly just want to call fit()
        to generate training data that can be used with some other options.
        '''

        task = self.env.taskModel()
        if not task.compiled:
            raise RuntimeError('environment must have associated compiled task model!')

        policies = ContinuousSamplerTaskPolicies(task)
        search = RandomSearch(policies)

        for _ in xrange(self.iter):

            action = None
            control = None

            while True:

                node = Node(world=self.env.world, root=True)

                if action is None:
                    action = policies.sample(node)
                    print action.tag
            
                if action.condition(node.world,
                        node.state,
                        node.world.actors[0],
                        node.world.actors[0].last_state):
                    control = action.getAction(node)
                else:
                    action = None
                    control = None

                if control is not None:
                    features, reward, done, info = self.env.step(control)
                    self._addToDataset(action.tag, self.env.world, control, features, reward, done)

        return None
        
