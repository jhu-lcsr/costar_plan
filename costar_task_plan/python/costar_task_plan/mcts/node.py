from costar_task_plan.abstract import *

'''
An MCTS node is a TYPE of state, but contains a different type of state.
Why? So that we can do interesting learning over MCTS states.
'''


class Node(AbstractState):

    next_idx = 0

    def __init__(self, world=None, action=None, prior=1., root=False):
        if world is None and action is None:
            raise RuntimeError('must provide either a world or an action!')

        self.parent = None
        self.n_visits = 0
        self.n_rollouts = 0
        self.world = world
        self.action = action
        self.children = []
        self.max_reward = -float('inf')
        self.total_reward = 0
        self.avg_reward = 0
        self.prev_reward = 0
        self.max_final_reward = -float('inf')
        self.prior = prior
        self.initialized = self.world is not None
        self.terminal = self.world is not None and self.world.done
        if self.action is not None and self.action.tag is not None:
            self.tag = self.action.tag
        elif root:
            self.tag = 'ROOT()'
        else:
            self.tag = 'NODE'
        if self.world is not None:
            self.state = self.world.actors[0].state
        else:
            self.state = None

        self.traj = []
        self.rewards = []
        self.reward = 0

    '''
    MCTS update step
    '''

    def update(self, reward, final_reward, steps):
        self.total_reward += reward
        self.max_reward = max(reward, self.max_reward)
        self.max_final_reward = max(reward, self.max_final_reward)
        self.avg_reward = self.total_reward / self.n_visits
        self.reward = 0  # reset counter for this trace

    '''
    expand() creates a new child world from a particular environment. We can
    then tick() this world forward however long we want to follow a particular
    policy.
    '''

    def expand(self, action):
        if action is None:
            raise RuntimeError('Cannot expand with an empty action!')
        if not isinstance(action, AbstractAction):
            raise TypeError(
                'node.expand() takes an Environment action, not an MCTS action.')
        new_world = self.world.fork(action)
        n = Node(world=new_world)
        n.prev_reward = self.prev_reward + self.reward
        n.parent = self
        return n

    '''
    Alternate version:
    - if all of our children have been created ahead of time, we may want to
    '''

    def instantiate(self, child):

        if child.parent is None:
            child.parent = self
        elif child.parent is not self:
            raise RuntimeError(
                'Cannot instantiate node with a node other than its parent!')

        if child.world is None:
            if not isinstance(child, Node):
                raise RuntimeError(
                    'Cannot instantiate someting that is not an MCTS node.')
            if child.action is None:
                raise RuntimeError(
                    'Cannot instantiate a node with an empty action!')

            action = child.action.getAction(self)
            if action is None:
                failed = True
            else:
                failed = False

            new_world = self.world.fork(action)
            child.world = new_world
            child.state = child.world.actors[0].state
            child.initialized = True
            child.terminal = child.world.done or failed
            child.rewards = [new_world.initial_reward]
            child.reward += new_world.initial_reward
            child.parent = self
            child.prev_reward = self.prev_reward + self.reward
            child.traj.append((self.world.actors[0].state, action))
            child.action.update(child)
        else:
            raise RuntimeError(
                'Cannot instantiate a node that already has been instantiated!')

    @property
    def ticks(self):
        return self.world.ticks

    '''
    tick() to advance the state of the world associated with a particular
    action or policy.
    '''

    def tick(self, action):
        if len(self.children) is not 0:
            raise RuntimeError(
                'Cannot tick a node that has already been expanded!')
        elif self.terminal:
            raise RuntimeError('Tried to advance from a terminal state!')
        else:

            if action is None:
                failed = True
            else:
                failed = False

            # advance the current state of the world
            (res, S0, A0, S1, F1, r) = self.world.tick(action)

            self.traj.append((S0, A0))
            self.rewards.append(r)
            self.reward += r
            self.terminal = self.world.done or failed
            self.state = S1

            return res and not failed, S0, A0, S1, F1, r

    ''' -----------------------------------------------------------------------
    NOTE: these are helper functions for accessing and updating the world state
    ----------------------------------------------------------------------- '''

    def features(self):
        if self.world is None:
            raise RuntimeError('node.instantiate() not called yet!')
        return self.world.initial_features
