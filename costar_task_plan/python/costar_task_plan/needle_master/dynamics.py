from costar_task_plan.abstract import AbstractDynamics
from actor import *

class NeedleMasterDynamics(AbstractDynamics):
    
  def apply(self, state, action):

    # convert action from params into trajectory
    action.finalize()

    # get that trajectory
    return NeedleState(action.state_traj[-1])

class NeedleControlDynamics(AbstractDynamics):

  def apply(self, pt, control):
      w = pt.w() + control.dw()
      x = pt.x() + control.v() * np.cos(w)
      y = pt.y() + control.v() * np.sin(w)
      pt = NeedleState(self.world, np.array([x,y,w]))
      return pt

class NeedleDynamics(AbstractTrajectoryDynamics):

  def __init__(self, world):
    self.world = world
    self.control_dynamics = NeedleControlDynamics(world)

  def apply(self, state, action):
    next_state = super(NeedleDynamics,self).apply(state, action)

    return NeedleTrajectory(next_state.traj)
