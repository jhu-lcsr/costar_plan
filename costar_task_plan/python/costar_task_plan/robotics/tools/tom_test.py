#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from costar_task_plan.robotics.core import *
from costar_task_plan.robotics.tom import TomWorld
from costar_task_plan.abstract import AbstractReward

import rospy

def load_tom_world():
  return TomWorld('./')

class NullReward(AbstractReward):
    def __call__(self, world, *args, **kwargs):
        return 0., 0.
    def evaluate(self, world, *args, **kwargs):
        return 0., 0.

'''
Main function:
  - create a TOM world
  - verify that the data is being managed correctly
  - fit models to data
  - create Reward objects as appropriate
  - create Policy objects as appropriate
'''
def load_tom_data_and_run():

  import signal
  signal.signal(signal.SIGINT, exit)

  try:
    rospy.init_node('tom_test_node')
    world = load_tom_world()
    world.makeRewardFunction("box")
    world.reward = NullReward()
  except RuntimeError, e:
    print "Failed to create world. Are you in the right directory?"
    raise e

  rate = rospy.Rate(1)
  try:
    while True:
      world.tick(world.zeroAction())
      rate.sleep()
  except rospy.ROSInterruptException, e:
    pass

if __name__ == "__main__":
  load_tom_data_and_run()



