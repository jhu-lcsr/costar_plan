
# This is a plan for executing a predefined sequence of options.
class ExecutionPlan(object):

    def __init__(self, plan, verbose=True):
        self.policies = []
        self.conditions = []

        if verbose:
            print "Creating execution plan from tree search results:"
        for node in plan:
            print "node =", node.tag, 
            if node.action is None:
                raise RuntimeError('execution plan must contain a whole sequence')
            if node.action.policy is None:
                raise RuntimeError('execution plan must contain a whole sequence of policies')
            if node.action.condition is None:
                raise RuntimeError('execution plan must contain a whole sequence of conditions')
            print "action =", action.tag
            
            self.policies.append(node.action.policy)
            self.conditions.append(node.action.condition)

        self.reset()

    # set counters to zero
    def reset(self):
        self.idx = 0
        self.world = None

    
    def step(self, world):
        # if we are advancing along a trajectory get the next action...

        # if we are done, increase our index
