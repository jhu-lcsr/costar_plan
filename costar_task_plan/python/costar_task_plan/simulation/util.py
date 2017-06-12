
from tasks import *
from robots import *


def GetAvailableTasks():
    return ["blocks", "clutter", "mug", "sorting", "explore", "oranges"]


def GetAvailableRobots():
    return ["jaco", "ur5_2_finger", "iiwa_3_finger"]


def GetAvailableAlgorithms():
    return [None, "ddpg", "cdqn"]


def GetTaskDefinition(task, robot, *args, **kwargs):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            'blocks': BlocksTaskDefinition(robot, *args, **kwargs),
            'clutter': ClutterTaskDefinition(robot, *args, **kwargs),
            'mug': MugTaskDefinition(robot, *args, **kwargs),
            'sorting': SortingTaskDefinition(robot, *args, **kwargs),
            'explore': ExploreTaskDefinition(robot, *args, **kwargs),
            'rings': RingsTaskDefinition(robot, *args, **kwargs),
            'oranges': OrangesTaskDefinition(robot, *args, **kwargs),
        }[task]
    except KeyError, e:
        raise NotImplementedError('Task %s not implemented!' % task)


def GetRobotInterface(robot, *args, **kwargs):
    '''
    Returns a simple interface representing a particular robot in the
    simulation.
    '''
    try:
        return {
            'ur5_2_finger': Ur5RobotiqInterface(*args, **kwargs),
            'iiwa_3_finger': IiwaRobotiq3FingerInterface(*args, **kwargs),
            'jaco':JacoRobotiqInterface(*args, **kwargs),
        }[robot]
    except KeyError, e:
        raise NotImplementedError('Robot %s not implemented!' % robot)
