'''
By Chris Paxton
Copyright (c) 2017, The Johns Hopkins University
See license for details.
'''

class AbstractExecute(object):
    '''
    This is the simple callable that sends the command to be executed to whatever
    robot, simulation, etc. that actually performs the commands and gets the next
    world state.
    '''

    def __call__(self, cmd):
        raise RuntimeError('Execute function was not provided.')

class DefaultExecute(AbstractExecute):
    '''
    This variant on the basic execute function will just tick a world.
    '''

    def __init__(self, world):
        self.world = world

    def __call__(self, cmd):
        self.world.tick(cmd)

class AbstractExecutionManager(object):
    '''
    Basic class for running execution of a sequence of actions forward in time.
    This will follow policies according to however it is that you actually want
    to execute them.
    '''

    def __init__(self, execute_fn, verbose, actor_id):
        self.execute = execute_fn
        self.verbose = verbose
        self.actor_id = actor_id

        self.reset(actor_id)

    def reset(self, actor_id=0):
        '''
        set counters to zero
        '''
        self.idx = 0
        self.actor_id = actor_id

    def step(self, world):
        '''
        Advance the state of the world once considering the current list of
        options for execution.
        '''
        cmd = self.apply(world)
        # Call our execute function. In general this will either:
        #   (a) send a command to a robot or a simulation running in another
        #       thread; or
        #   (b) tick the world and possibly display test metrics.
        if cmd is not None:
            self.execute(cmd)
            return False
        else:
            return True
        return world.done

class PlanExecutionManager(AbstractExecutionManager):
    '''
    This is a plan for executing a predefined sequence of options. It requires:
    - a plan -- a sequence of nodes
    - an execute function -- this actually takes the commands and turns them into
    observable behavior.
    '''

    def __init__(self, plan, execute_fn=AbstractExecute(), verbose=True, actor_id=0):
        super(PlanExecutionManager, self).__init__(execute_fn, verbose, actor_id)
        self.nodes = []

        print "Creating execution plan from tree search results:"
        for i, node in enumerate(plan):
            print "node =", node.tag,

            # check node
            if node.action is None and i == 0:
                print " -- skipping root"
                continue
            elif node.action is None:
                raise RuntimeError(
                    'execution plan must contain a whole sequence of actions')

            # check policy
            if node.action.policy is None:
                raise RuntimeError(
                    'execution plan must contain a whole sequence of policies')

            # check condition
            if node.action.condition is None:
                raise RuntimeError(
                    'execution plan must contain a whole sequence of conditions')

            print "action =", node.action.tag
            self.nodes.append(node)

    def apply(self, world):
        '''
        Return the next command that we should be applying according to this task
        plan.

        Returns None if no command to execute.
        '''
        actor = world.actors[self.actor_id]
        state = actor.state
        cmd = None
        done = world.done

        # Check to see if we can even be executing at this point. If not, then
        # we should just quit. This also applies if we somehow blundered into
        # a terminal state.
        if done:
            return None

        # Main loop. Check each action and see if it's still active. If it is
        # not active any more, then move on to the next action. Get the control
        # associated with the newest action that you found.
        while cmd is None and self.idx < len(self.nodes):
            # If we are advancing along a trajectory get the next action, as
            # long as that trajectory's gating conditions hold. Once they are
            # no longer true, then we can try to move on to the next trajectory
            # (or action, or option; use your preferred nomenclature here).
            condition = self.nodes[self.idx].action.condition
            if condition(world, state, actor, actor.last_state):
                policy = self.nodes[self.idx].action.policy
                cmd = policy.evaluate(world, state, actor)
            else:
                cmd = None
                self.idx += 1

        # If we could not come up with a set of controls to execute, execute
        # the default fallback action. For us this is just a zero motion.
        if cmd is None:
            cmd = world.zeroAction(0)
            done = True
        return cmd


class OptionsExecutionManager(AbstractExecutionManager):

    '''
    This class is the cousin of the above: it works by executing a set of
    non-instantiated options. Whenever a new option is reached it will sample a
    policy as appropriate and create the necessary termination conditions.
    '''

    def __init__(self, options, execute_fn=AbstractExecute(), verbose=True,
            actor_id=0):
        super(OptionsExecutionManager, self).__init__(execute_fn, verbose, actor_id)
        self.options = options
        self.condition = None
        self.policy = None

    def reset(self, actor_id=0):
        '''
        set counters to zero
        '''
        super(OptionsExecutionManager,self).reset(self.actor_id)
        self.condition = None
        self.policy = None

    def apply(self, world):
        '''
        Return the next command that we should be applying according to this task
        plan.

        Returns None if no command to execute.
        '''
        actor = world.actors[self.actor_id]
        state = actor.state
        cmd = None

        # Check to see if we can even be executing at this point. If not, then
        # we should just quit. This also applies if we somehow blundered into
        # a terminal state.
        if world.done:
            return None

        cmd = None
        # If we are currently following some policy...
        # Then we want to check its associated condition and increment the
        # counter if possible.
        while cmd is None and self.idx < len(self.options):
            # If we do not have a policy right now, we want to sample one to follow
            # for the time being.
            if self.policy is None:
                self.policy, self.condition = self.options[self.idx].samplePolicy(world)

            if self.condition(world, state, actor, actor.last_state):
                cmd = self.policy.evaluate(world, state, actor)
            else:
                # Move to the next option.
                if not self.options[self.idx].checkPostcondition(world, state):
                    raise RuntimeError('asdf')
                    return None
                self.idx += 1
                self.condition = None
                self.policy = None
                print self.idx
        
        return cmd
