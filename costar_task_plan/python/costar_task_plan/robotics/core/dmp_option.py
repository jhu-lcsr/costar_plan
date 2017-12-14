
from costar_task_plan.abstract import AbstractOption, AbstractCondition
from costar_task_plan.robotics.representation import CartesianSkillInstance
from dmp_policy import JointDmpPolicy, CartesianDmpPolicy

import numpy as np

SAMPLING_MODE_NORMAL = 0
SAMPLING_MODE_LIST = 1

class DmpOption(AbstractOption):
    '''
    '''

    def __init__(self,
            config, # robot config file
            kinematics, # kinematics of the robot 
            goal_object, # type of object to arrive at 
            skill_name, # name of this skill
            feature_model, # feature model
            policy_type=CartesianDmpPolicy, # what kind of DMP are we creating
            in_hand_object=None,
            sampling_mode=SAMPLING_MODE_NORMAL,
            traj_dist=None,):

        '''
        Represent information about the robot to create DMP trajectories. This
        class allows us to sample reasonable trajectories from a set of provided
        demonstrations, represented as a Cartesian skill model.

        Parameters:
        -----------
        config: dict of robot configuration options
        kinematics: one or two armed kinematics models for the robot
        policy_type;
        '''

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

        self.config = config
        self.goal = goal_object
        self.policy_type = policy_type
        self.kinematics = kinematics
        self.feature_model = feature_model
        self.skill_name = skill_name
        self.traj_dist = traj_dist
        self.in_hand = in_hand_object
        self.skill_instance = CartesianSkillInstance(self.config, self.traj_dist.mu)

    def makePolicy(self, *args, **kwargs):
        '''
        Deterministically select the first policy from the list.
        '''
        return self.policy_type(
            skill=self,
            goal=self.goal,
            dmp=self.skill_instance,
            kinematics=self.kinematics), DmpCondition(
            parent=self,
            goal=self.goal,
            kinematics=self.kinematics,)

    def samplePolicy(self, *args, **kwargs):
        '''
        Randomly create a policy
        '''
        if self.traj_dist is None:
            raise RuntimeError('Attempted to sample from a mis-specified'
                    ' action!')
        if self.sampling_mode = SAMPLING_MODE_NORMAL:
            params = np.random.multivariate_normal(
                    self.traj_dist.mu,
                    self.traj_dist.sigma)
            skill_instance = CartesianSkillInstance(config=self.config, params=params)
        elif self.sampling_mode = SAMPLING_MODE_LIST:
            
        return self.policy_type(
            skill=self,
            goal=self.goal,
            dmp=skill_instance,
            kinematics=self.kinematics), DmpCondition(
            parent=self,
            goal=self.goal,
            kinematics=self.kinematics,)

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
