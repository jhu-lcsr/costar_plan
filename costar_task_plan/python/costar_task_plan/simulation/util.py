
from tasks import *
from robots import *


def GetAvailableTasks():
    return ["blocks", "drl_blocks", "clutter", "sorting", "tower", "oranges", "obstacles", "trays", "obstructions", "sorting"]


def GetAvailableRobots():
    return ["jaco", "ur5_2_finger", "ur5", "ur5_robotiq"]


def GetAvailableAlgorithms():
    return [None, "ddpg", "cdqn"]


def GetTaskDefinition(task, robot, features, *args, **kwargs):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            'blocks': lambda: BlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'trays': lambda: BlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'obstacles': lambda: ObstaclesTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'obstructions': lambda: ObstructionsTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'drl_blocks': lambda: DRLBlocksTaskDefinition(0, robot=robot, features=features, *args, **kwargs),
            'tower': lambda: BlocksTaskDefinition(None, robot, features=features, *args, **kwargs),
            'clutter': lambda: ClutterTaskDefinition(robot, features=features, *args, **kwargs),
            'sorting': lambda: SortingTaskDefinition(robot, features=features, *args, **kwargs),
            'oranges': lambda: OrangesTaskDefinition(robot, features=features, *args, **kwargs),
        }[task.lower()]()
    except KeyError, e:
        raise NotImplementedError('Task %s not implemented!' % task)


def GetRobotInterface(robot, *args, **kwargs):
    '''
    Returns a simple interface representing a particular robot in the
    simulation.
    '''
    try:
        return {
            'ur5_2_finger': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'ur5_robotiq': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'ur5': lambda: Ur5RobotiqInterface(*args, **kwargs),
            'jaco': lambda: JacoRobotiqInterface(*args, **kwargs),
            #'iiwa_3_finger': IiwaRobotiq3FingerInterface(*args, **kwargs),
            #'iiwa': IiwaRobotiq3FingerInterface(*args, **kwargs),
        }[robot.lower()]()
    except KeyError, e:
        raise NotImplementedError('Robot %s not implemented!' % robot)
