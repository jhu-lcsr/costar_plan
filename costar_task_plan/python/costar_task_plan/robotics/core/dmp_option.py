
from dmp_policy import JointDmpPolicy, CartesianDmpPolicy

import numpy as np

from costar_task_plan.abstract import AbstractOption, AbstractCondition


class DmpOption(AbstractOption):

    def __init__(self, policy_type, kinematics, goal, skill_name, model, instances=[], attached_frame=None):
        if isinstance(policy_type, str):
            # parse into appropriate constructor
            if policy_type == 'joint':
                policy_type = JointDmpPolicy
                raise NotImplementedError(
                    'Joint space skills not currently implemented.')
            elif policy_type == 'cartesian':
                policy_type = CartesianDmpPolicy
            else:
                raise RuntimeError(
                    'invalid option for DMP policy type: %s' % policy_type)
        if not isinstance(policy_type, type):
            raise RuntimeError('invalid data type for DMP policy')
        if attached_frame is not None:
            raise NotImplementedError('attached frame is not yet supported')

        self.goal = goal
        self.policy_type = policy_type
        self.kinematics = kinematics
        self.instances = instances
        self.model = model
        self.skill_name = skill_name
        self.attached_frame = attached_frame

    # Make a policy.
    def makePolicy(self, *args, **kwargs):
        return self.policy_type(
            skill=self,
            goal=self.goal,
            dmp=self.instances[0],
            kinematics=self.kinematics)

    def samplePolicy(self, *args, **kwargs):
        pass

    # Get the gating condition for a specific option.
    # - execution should continue until such time as this condition is true.
    def getGatingCondition(self, *args, **kwargs):
        return DmpCondition(
            parent=self,
            goal=self.goal,
            kinematics=self.kinematics,)

    # Is it ok to begin this option?
    def checkPrecondition(self, world, state):
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPrecondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPrecondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPrecondition() not yet implemented!')

    # Did we successfully complete this option?
    def checkPostcondition(self, world, state):
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPostcondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPostcondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPostcondition() not yet implemented!')


class DmpCondition(AbstractCondition):

    '''
    This condition tells us whether or not we successfully arrived at the end of
    an action. It is true while we should continue executing. If our ee pose is
    within tolerances and we are nearly stopped, it returns false.
    '''

    def __init__(self, parent, goal, kinematics):
        self.goal = goal
        self.parent = parent
        self.kinematics = kinematics

    def __call__(self, world, state, actor=None, prev_state=None):
        if actor is None:
            actor = world.actors[0]

        # Determine if we finished the last action so we can switch active
        # DMPs.
        ok_to_start = self.parent is not state.reference and \
            (state.finished_last_sequence or state.reference is None)
        return ok_to_start or np.any(np.abs(state.dq) > 1e-2) or state.seq > 0
