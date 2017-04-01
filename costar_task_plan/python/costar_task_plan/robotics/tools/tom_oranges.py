#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See license for more details

from costar_task_plan.abstract import Task
from costar_task_plan.robotics.core import DmpOption
from costar_task_plan.robotics.core import JointDmpPolicy, CartesianDmpPolicy
from costar_task_plan.robotics.tom import TomWorld
from costar_task_plan.robotics.tom import TomGripperOption, TomGripperCloseOption, TomGripperOpenOption
from costar_task_plan.mcts import *

# Set up the "pick" action that we want to performm
def __pick_args():
  return {
    "constructor": DmpOption,
    "args": ["orange","kinematics","policy_type"],
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

def __move_args():
  return {
    "constructor": DmpOption,
    "args": ["squeeze_area","kinematics","policy_type"],
    "remap": {"squeeze_area": "goal_frame"},
      }

def __test_args():
  return {
    "constructor": DmpOption,
    "args": ["squeeze_area","kinematics","policy_type"],
    "remap": {"squeeze_area": "goal_frame"},
      }

def __box_args():
  return {
    "constructor": DmpOption,
    "args": ["box","kinematics","policy_type"],
    "remap": {"box": "goal_frame"},
      }

def __trash_args():
  return {
    "constructor": DmpOption,
    "args": ["box","kinematics","policy_type"],
    "remap": {"box": "goal_frame"},
      }

# Instantiate the whole task model based on our data. We must make sure to
# provide the lfd object containing models, etc., or we will not properly
# create all of the different DMP models.
def MakeTomTaskModel():
  task = Task()
  task.add("pick", None, __pick_args())
  task.add("grasp1", ["pick"], __grasp_args())
  task.add("move", ["grasp1"], __move_args())
  task.add("release", ["move"], __release_args())
  task.add("test", ["release"], __test_args())
  task.add("grasp2", ["test"], __grasp_args())
  task.add("box", ["grasp2"], __box_args())
  task.add("trash", ["grasp2"], __trash_args())
  return task

if __name__ == '__main__':

  # Create the task model
  task = MakeTomTaskModel()
  world = TomWorld('./',load_dataset=False)

  # Set up arguments for tom sim task
  args = {
    'orange': ['orange1', 'orange2', 'orange3'],
    'squeeze_area': ['squeeze_area1'],
    'box': ['box1'],
    'trash': ['trash1'],
    'kinematics': [world.lfd.kdl_kin],
    'policy_type': [CartesianDmpPolicy],
  }

  # Create task definition
  args = task.compile(args)

  # Print out a summary of our task model. We then need to use this to create 
  # the MCTS data types that we are performing our search over.
  print task.nodeSummary()


