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
from costar_task_plan.tools import showGraph

# TOM ORANGES TASK
# This one defines the task we want to finish putting together for the
# different robots. It creates the world observation information -- which
# objects we can see -- and also the task structure.

# Set up the "pickup" action that we want to performm
def __pick_args(lfd):
  dmp_maker = __get_dmp_maker("pickup", lfd)
  return {
    "constructor": dmp_maker,
    "args": ["orange"],
    "remap": {"orange": "goal"},
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

def __move_args(lfd):
  dmp_maker = __get_dmp_maker("pickup", lfd)
  return {
    "constructor": dmp_maker,
    "args": ["squeeze_area"],
    "remap": {"squeeze_area": "goal"},
      }

def __test_args(lfd):
  dmp_maker = __get_dmp_maker("pickup", lfd)
  return {
    "constructor": dmp_maker,
    "args": ["squeeze_area"],
    "remap": {"squeeze_area": "goal"},
      }

def __box_args(lfd):
  dmp_maker = __get_dmp_maker("pickup", lfd)
  return {
    "constructor": dmp_maker,
    "args": ["box"],
    "remap": {"box": "goal"},
      }

def __trash_args(lfd):
  dmp_maker = __get_dmp_maker("pickup", lfd)
  return {
    "constructor": dmp_maker,
    "args": ["trash"],
    "remap": {"trash": "goal"},
      }


def __get_dmp_maker(skill_name,lfd):

  dmp_maker = lambda goal: DmpOption(
      goal=goal,
      instances=lfd.skill_instances[skill_name],
      model=lfd.skill_models[skill_name],
      kinematics=lfd.kdl_kin,
      policy_type=CartesianDmpPolicy)
  return dmp_maker

# Instantiate the whole task model based on our data. We must make sure to
# provide the lfd object containing models, etc., or we will not properly
# create all of the different DMP models.
def MakeTomTaskModel(lfd):

  task = Task()
  #task.add("pickup", None, __pick_args(lfd))
  #task.add("grasp1", ["pickup"], __grasp_args())
  #task.add("move", ["grasp1"], __move_args(lfd))
  task.add("move", None, __move_args(lfd))
  task.add("release1", ["move"], __release_args())
  task.add("test", ["release1"], __test_args(lfd))

  #task.add("grasp2", ["test"], __grasp_args())
  #task.add("box", ["grasp2"], __box_args(lfd))
  #task.add("trash", ["grasp2"], __trash_args(lfd))
  #task.add("release2", ["box", "trash"], __release_args())
  return task

# Set up arguments for tom sim task
def OrangesTaskArgs():
  args = {
    'orange': ['orange1', ],#'orange2', 'orange3'],
    'squeeze_area': ['squeeze_area1'],
    'box': ['box1'],
    'trash': ['trash1'],
  }
  return args

if __name__ == '__main__':

  # Create the task model
  world = TomWorld('./',load_dataset=False)
  task = MakeTomTaskModel(world.lfd)
  args = OrangesTaskArgs()

  # Create task definition
  filled_args = task.compile(args)

  # Print out a summary of our task model. We then need to use this to create 
  # the MCTS data types that we are performing our search over.
  print task.nodeSummary()

  print task.children['root()']

