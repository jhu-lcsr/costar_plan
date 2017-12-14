from costar_task_plan.abstract import *

import numpy as np

'''
Tracks the player's progress in the game so far
'''
class NeedleState(AbstractState):
  vec = np.array([0,0,0])
  def __init__(self,env,vec):
    self.vec = vec
    self.gates = [False]*len(env.gates())

  def x(self):
    return self.vec[0]

  def y(self):
    return self.vec[1]

  def w(self):
    return self.vec[2]

'''
Wrap StateTrajectory to help draw
'''
class NeedleTrajectory(StateTrajectory):

  def __init__(self, *args, **kwargs):
    super(NeedleTrajectory, self).__init__(*args, **kwargs)
    # for easy access
    tmp = []
    for pt in self.traj:
      tmp.append([pt.x(), pt.y(), pt.w()])
    self.s = np.array(tmp)

  def show(self):
    import matplotlib.pyplot as plt
    plt.plot(self.s[:, 0], self.s[:, 1])

'''
Control: an action consists of a whole sequence of these
'''
class NeedleControl(AbstractAction):

  def __init__(self,v,w):
    self.vec = np.array([v,w])

  def v(self):
    return self.vec[0]
  def dw(self):
    return self.vec[1]

'''
Needle Master motions are sets of 2D arcs.
'''
class NeedleAction(AbstractTrajectoryAction):

  def __init__(self, primitives=3, params=[]):
    self.primitives = primitives
    self.params = params
    self.traj = []

  # note: override this
  def _finalize(self, state):
    for i in xrange(self.primitives):
      v = self.params[3*i]
      dw = self.params[3*i + 1]
      t = self.params[3*i + 2]
      for j in xrange(int(t)):
        self.traj.append(NeedleControl(v,dw))

'''
The actor in the needle game is the only player
'''
class NeedleActor(AbstractActor):
  def __init__(self, state, policy=None):
    super(NeedleActor, self).__init__(policy=policy)
    self.state = state
