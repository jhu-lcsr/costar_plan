from __future__ import print_function

from costar_task_plan.abstract.task import *


def GetPoses():
    '''
    All poses have been recorded relative to /base_link. If the robot moves
    they may no longer work.
    '''
    pose1_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector(0.493, -0.202, 0.216))
    pose2_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose3_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose4_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose1_right = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector(0.493, -0.202, 0.216))
    pose2_right = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose3_right = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose4_right = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    pose_home = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector())
    poses = {"home": pose_home,
             "pose1_left": pose1_left,
             "pose2_left": pose2_left,
             "pose3_left": pose3_left,
             "pose4_left": pose4_left,
             "pose1_right": pose1_right,
             "pose2_right": pose2_right,
             "pose3_right": pose3_right,
             "pose4_right": pose4_right,}
    return poses


def MakeStackTask():
    '''
    Create a version of the robot task for stacking two blocks.
    '''

    task = Task()

    # Create sub-tasks for left and right
    pickup_left = _makePickupLeft()
    pickup_right = _makePickupRight()
    place_left = _makePlaceLeft()
    place_right = _makePlaceRight()

    # Create the task: pick up any one block and put it down in a legal
    # position somewhere on the other side of the bin.
    task = Task()
    task.add("pickup_left", None, pickup_left)
    task.add("pickup_right", None, pickup_right)
    task.add("place_left", "pickup_right", place_left)
    task.add("place_right", "pickup_left", place_right)
    task.add("DONE", ["place_right", "place_left"], {})

    return task

def _makePickupLeft():
    pickup = TaskTemplate("pickup_left", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePickupRight():
    pickup = TaskTemplate("pickup_right", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePlaceLeft():
    place = TaskTemplate("place_left", ["pickup_right"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _makePlaceRight():
    place = TaskTemplate("place_right", ["pickup_left"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _pickupLeftArgs():
    # Create args for pickup from left task
    return {
        "task": pickup_left,
        "args": ["block1"],
    }

def _pickupRightArgs():
    # And create args for pickup from right task
    return {
        "task": pickup_right,
        "args": ["block1"],
    }

def _homeArgs():
    return {}

def _detectObjectsArgs():
    return {}

def _checkBlocks1And2(block1,block2,**kwargs):
    '''
    Simple function that is passed as a callable "check" when creating the task
    execution graph. This makes sure we don't build branches that just make no
    sense -- like trying to put a blue block on top of itself.

    Parameters:
    -----------
    block1: unique block name, e.g. "red_block"
    block2: second unique block name, e.g. "blue_block"
    '''
    return not block1 == block2
