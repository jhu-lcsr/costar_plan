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

    def __init__(self, iter=10000, *args, **kwargs):
        super(TaskAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):

        task = env.taskModel()
        if not task.compiled:
            raise RuntimeError('environment must have associated compiled task model!')

        policies = ContinuousSamplerTaskPolicies(task)
        search = RandomSearch(policies)

        while True:

            action = None
            control = None
            node = Node(world=env.world, root=True)

            if action is None:
                action = policies.sample(node)
                print "sampled ", action, action.condition
        
            if action.condition(node.world,
                    node.state,
                    node.world.actors[0],
                    node.world.actors[0].last_state):
                control = action.getAction(node)
            else:
                action = None

            if control is not None:
                print "sending command",
                print control.cmd
                env.step(control.cmd)

        return None
        
