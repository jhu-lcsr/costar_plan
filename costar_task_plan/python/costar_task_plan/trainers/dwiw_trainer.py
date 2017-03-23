# By Chris Paxton
# (c) 2017 Johns Hopkins University
# See license for details.

from abstract import AbstractTrainer


# This is the "Do-what-I-want" trainer, expanded from the earlier paper.
# The idea is that this one assumes we have:
# - a controller (DMP, for example)
# - a current state
# - and a goal.
# We want to learn which controllers to try for different goals, so that they
# stick as close as possible to our reward function.
class DwiwTrainer(AbstractTrainer):

  def __init__(self, env, task):
    pass
