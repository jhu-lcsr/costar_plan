from dynamics import *
import numpy as np

class AbstractTrajectoryAction(AbstractAction):

  state_traj = [] # states
  control_traj = [] # controls

  _finalized = False

  def finalize(self, state):
    if self._finalized:
      return False
    else:
      self._finalized = True

      # this has to be overidden
      self._finalize(state)
      return True


  # problem specific implementation
  def _finalize(self):
    raise Exception('action._finalize() not implemented!')

class AbstractTrajectoryDynamics(AbstractDynamics):

  def __init__(self, world):
    super(AbstractTrajectoryDynamics, self).__init__(world)
    self.control_dynamics = None

  def apply(self, state, action):
    assert(self.control_dynamics is not None)
    action.finalize(state)
    traj = []
    pt = state.end()
    for control in action.traj:
      pt = self.control_dynamics(pt, control)
      traj.append(pt)

    return StateTrajectory(traj)

'''
tracks a whole trajectory of actions
'''
class StateTrajectory(AbstractState):

  def __init__(self, traj=[]):
    self.traj = traj
    self.predicates = []

  def begin(self):
    return self.traj[0]

  def end(self):
    return self.traj[-1]

  def updatePredicates(self, world, actor):
    if actor.last_state is not None:
      self.traj[0].predicates = [check(world, self.traj[0], actor, actor.last_state.end())
          for (name, check)
          in world.predicates]
    else:
      self.traj[0].predicates = [check(world, self.traj[0], actor)
          for (name, check)
          in world.predicates]
    for i in xrange(len(self.traj)-1):
      self.traj[i+1].predicates = [check(world, self.traj[i+1], actor, self.traj[i])
          for (name, check)
          in world.predicates]
