from costar_task_plan.abstract.task import *

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
    place_args = {
        "task": place,
        "args": ["block2"],
    }
    task = Task()
    task.add("pickup_left", None, _pickupLeftArgs())
    task.add("place_right", "pickup_left", place_args)
    task.add("pickup_right", None, pickup_args)
    task.add("place_left", "pickup_right", place_args)
    task.add("DONE", ["place_right", "place_left"], place_args)

def _makePickupLeft():
    pickup = TaskTemplate("pickup_left", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return pickup

def _makePlace():
    place = TaskTemplate("place", "pickup")
    return place

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
