#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See license for more details

from costar_task_plan.robotics.core import DmpOption
from costar_task_plan.robotics.tom import TomGripperOption, TomGripperCloseOption, TomGripperOpenOption

# Set up the "pick" action that we want to performm
def __pick_args():
  return {
    "constructor": DmpOption,
    "args": ["orange","kinematics"],
    "remap": {"orange": "goal_frame"},
      }

def __grasp_args():
  return {
    "constructor": TomGripperCloseOption,
    "args": [],
      }

def __release_args():
  return {
    "constructor": TomGripperOpenOption,
    "args": [],
      }



# Instantiate the whole task model based on our data. We must make sure to
# provide the lfd object containing models, etc., or we will not properly
# create all of the different DMP models.
def MakeTomTaskModel(lfd):
  task = Task()
  task.add("pick", None, __pick_args())
  task.add("grasp1", ["pick"], __grasp_args())
  task.add("move", ["grasp"], __move_args())
  task.add("release", ["move"], __move_args())
  task.add("test", None, __test_args())
  task.add("grasp2", ["test"], __grasp_args())
  task.add("box", ["grasp2"], __box_args())
  task.add("trash", ["grasp2"], __box_args())


