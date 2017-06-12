
from costar_task_plan.abstract import AbstractOption, AbstractCondition

class GoalDirectedMotionOption(AbstractOption):
    '''
    This represents a goal that will move us somewhere relative to a particular
    object, the "goal."

    This lets us sample policies that will take steps towards the goal. Actions
    are represented as a change in position; we'll assume that the controller
    running on the robot will be enough to take us to the goal position.
    '''

    def __init__(self, world, goal, pose=None, *args, **kwargs):
        self.goal_id = world.getObjectId[goal]
        if pose is not None:
            self.position, self.rotation = pose

    def makePolicy(self):
        pass

    def samplePolicy(self):
        pass

    def getGatingCondition(self, *args, **kwargs):
        # Get the gating condition for a specific option.
        # - execution should continue until such time as this condition is true.
        raise NotImplementedError('not supported')

    def checkPrecondition(self, world, state):
        # Is it ok to begin this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPrecondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPrecondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPrecondition() not yet implemented!')

    def checkPostcondition(self, world, state):
        # Did we successfully complete this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPostcondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPostcondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPostcondition() not yet implemented!')


class GeneralMotionOption(AbstractOption):
    '''
    This motion is not parameterized by anything in particular. This lets us 
    sample policies that will take us twoards this goal. 
    '''
    def __init__(self, pose, *args, **kwargs):
        if pose is not None:
            self.position, self.rotation = pose
