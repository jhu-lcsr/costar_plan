
from tasks import *
from robots import *

def GetAvailableTasks():
    return ["blocks"]

def GetAvailableRobots():
    return ["ur5_robotiq"]

def GetAvailableAlgorithms():
    return [None, "ddpg", "cdqn"]

def GetTaskDefinition(task, robot, *args, **kwargs):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
                'blocks': BlocksTaskDefinition(robot, *args, **kwargs),
        }[task]
    except KeyError, e:
        raise NotImplementedError('Task %s not implemented!'%task)

def GetRobotInterface(robot, *args, **kwargs):
    '''
    Returns a simple interface representing a particular robot in the
    simulation.
    '''
    try:
        return {
                'ur5_robotiq': Ur5RobotiqInterface(*args, **kwargs),
        }[robot]
    except KeyError, e:
        raise NotImplementedError('Robot %s not implemented!'%robot)
