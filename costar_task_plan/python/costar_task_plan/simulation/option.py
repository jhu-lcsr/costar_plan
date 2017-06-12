
from costar_task_plan.abstract import AbstractOption, AbstractCondition

class GoalDirectedMotionOption(AbstractOption):
    '''
    This represents a goal that will move us somewhere relative to a particular
    object, the "goal."

    This lets us sample policies that will take steps towards the goal. Actions
    are represented as a change in position; we'll assume that the controller
    running on the robot will be enough to take us to the goal position.
    '''

    def __init__(self, world, goal, pose, *args, **kwargs):
        self.goal_id = world.getObjectId[goal]
        self.position, self.rotation = pose


class GeneralMotionOption(AbstractOption):
    '''
    This motion is not parameterized by anything in particular. This lets us 
    sample policies that will take us twoards this goal. 
    '''
    def __init__(self, pose, *args, **kwargs):
        self.pose = pose
