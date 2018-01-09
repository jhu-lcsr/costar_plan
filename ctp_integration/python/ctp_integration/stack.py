from costar_task_plan.abstract.task import *

def MakeStackTask():
    '''
    Create a version of the robot task for stacking two blocks.
    '''

    task = Task()

    pickup = _makePickup()
    place = _makePlace()

    # Create the task: pick up any one block and put it down in a legal
    # position somewhere on the other side of the bin.
    pickup_args = {
        "task": pickup,
        "args": ["block1"],
    }
    place_args = {
        "task": place,
        "args": ["block2"],
    }
    task = Task()
    task.add("pickup", None, pickup_args)
    task.add("place", None, place_args)

def _makePickup():
    pickup = TaskTemplate("pickup", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return pickup

def _makePlace():
    place = TaskTemplate("place", "pickup")
    return place

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
