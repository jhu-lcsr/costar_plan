from costar_task_search.abstract import *

from environment import *
from demo import *
from actor import *

import matplotlib.pyplot as plt
import numpy as np

class NeedleMasterWorld(AbstractWorld):

  def __init__(self, filename, trials, randomize=True, start_from_trial=True):
    self.randomize=True

    # load the environment
    self.env = Environment(filename)
    self.trials = []
    for trial in trials:
      self.trials.append(Demo(env_height=self.env.height, env_width=self.env.width, filename=trial))

  def show(self, show_plot=False):
    self.env.Draw()
    for trial in self.trials:
      trial.Draw()
    if show_plot:
      plt.show()

  def sampleStart(self, from_trial=True):
    if from_trial:
      idx = np.random.randint(len(self.trials))
      return NeedleState(self, vec=self.trials[idx].s[0])
    else:
      raise Exception('drawing at random not yet implemented')

  def _initActor(self, from_trial=True):
    s0 = self.sample_start(from_trial=from_trial)
    self.actors[0] = NeedleActor(state=NeedleTrajectory[s0])

  def getLearner(self):
    return self.actors[0]

  def gates(self):
    return self.env.gates
