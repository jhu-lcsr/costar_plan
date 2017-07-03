'''
By Chris Paxton
Copyright (c) 2017, The Johns Hopkins University
See license for details.
'''

# This is the simple callable that sends the command to be executed to whatever
# robot, simulation, etc. that actually performs the commands and gets the next
# world state.
class AbstractExecute(object):
  def __call__(self, cmd):
    raise RuntimeError('Execute function was not provided.')

# This variant on the basic execute function will just tick a world.
class DefaultExecute(AbstractExecute):
  def __init__(self, world):
    self.world = world

  def __call__(self, cmd):
    self.world.tick(cmd)

# This is a plan for executing a predefined sequence of options. It requires:
# - a plan -- a sequence of nodes
# - an execute function -- this actually takes the commands and turns them into
#   observable behavior.
class ExecutionPlan(object):

  def __init__(self, plan, execute_fn=AbstractExecute(), verbose=True, actor_id=0):
    self.nodes = []
    self.execute = execute_fn

    print "Creating execution plan from tree search results:"
    for i, node in enumerate(plan):
      print "node =", node.tag, 
      
      # check node
      if node.action is None and i == 0:
          print " -- skipping root"
          continue
      elif node.action is None:
          raise RuntimeError('execution plan must contain a whole sequence of actions')

      # check policy
      if node.action.policy is None:
          raise RuntimeError('execution plan must contain a whole sequence of policies')

      # check condition
      if node.action.condition is None:
          raise RuntimeError('execution plan must contain a whole sequence of conditions')

      print "action =", node.action.tag
      self.nodes.append(node)

    self.reset(actor_id)

  # set counters to zero
  def reset(self, actor_id=0):
    self.idx = 0
    self.world = None
    self.actor_id = actor_id

  def apply(self, world):
    '''
    Return the next command that we should be applying according to this task
    plan.
    '''
    actor = world.actors[self.actor_id]
    state = actor.state
    cmd = None
    done = world.done

    # Check to see if we can even be executing at this point. If not, then
    # we should just quit. This also applies if we somehow blundered into
    # a terminal state.
    if done:
      return True

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

  def step(self, world):
    cmd = self.apply(world)
    # Call our execute function. In general this will either:
    #   (a) send a command to a robot or a simulation running in another
    #       thread; or
    #   (b) tick the world and possibly display test metrics.
    self.execute(cmd)
    return done

class ExecuteOptions(object):
    '''
    This class is the cousin of the above: it works by executing a set of
    non-instantiated options. Whenever a new option is reached it will sample a
    policy as appropriate and create the necessary termination conditions.
    '''
    pass

