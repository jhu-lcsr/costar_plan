from abstract import *

import numpy as np

'''
SimulationRollout takes as its sole argument.
'''


class SimulationRollout(AbstractRollout):

    def __init__(self, rollout_sample):
        if not isinstance(rollout_sample, AbstractSample):
            raise RuntimeError('Must provide an AbstractSample to rollout.')
        self.rollout_sample = rollout_sample

    def __call__(self, node, depth):
        reward = 0
        final_reward = 0
        steps_taken = 0
        i = 0
        if self.rollout_sample is not None:
            for i in xrange(depth):
                action = self.rollout_sample(node)
                node = action.apply(node)

                final_reward = node.reward
                reward += final_reward
                if node.terminal:
                    break
        steps_taken = i

        return reward, final_reward, steps_taken

'''
Depth First Search --
This class just continues to roll out the best path it can see until it gets to
its maximum horizon. Unlike in the Simulation rollout, this one actually adds
the nodes it visits to the tree, and does the whole backup procedure -- so we
need the appropriate policies.
'''


class DfsRollout(AbstractRollout):

    def __init__(self, policies):
        pass

    '''
  Take a node and follow the "best" actions all the way down the tree until we
  get to the end. This will actually initialize and set up the whole branch of
  the tree up until we hit a terminal state.
  '''

    def __call__(self, node, depth):
        pass

'''
Take as a value the action/value pair associated with a particular state.

Note that while it might make more sense to use a state value function here, we
are sometimes training with just an action/value function as a critic. This
means that we actually have the action/value function available already... so
let's try using that.
'''


class ActionValueRollout(AbstractRollout):

    def __init__(self, model, action_input, sample_action):
        self.model = model
        self.action_input = action_input
        self.sample_action = sample_action

    '''
  We actually don't care about depth for this one.
  '''

    def __call__(self, node, depth):

        # compute the default action we'd take from this node
        action = self.sample_action(node).getAction(node)
        inputs = []
        for i in self.model.input:
            if i == self.action_input:
                inputs.append(np.array([action.toArray().transpose()]))
            else:
                inputs.append(np.array([node.features()]))

        r = self.model.predict(inputs)[0][0]
        return r, r, 0
