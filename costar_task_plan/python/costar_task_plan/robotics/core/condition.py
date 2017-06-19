from costar_task_plan.abstract import AbstractCondition

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

import rospy


class ValidStateCondition(AbstractCondition):

    '''
    Check to see if a particular state is valid.

    TODO: make this more efficient by taking a reference to the C++ object that
    checks for collisions.
    '''

    def __init__(self):
        rospy.wait_for_service('check_state_validity', 5.0)
        self.srv = rospy.ServiceProxy('check_state_validity', GetStateValidity)

    def __call__(self, world, state, actor, prev_state=None):
        js = JointState(
            name=actor.joints,
            position=state.q,
        )
        rs = RobotState(joint_state=js)

        # TODO(cpaxton): this is about 75% of the current speed. Commenting
        # this out makes a huge difference. We should be able to drastically
        # reduce it by adding a C++ interface to run the check.
        res = self.srv(robot_state=rs)
        if not res.valid:
            print "[ValidStateCondition] Invalid state!"
        return res.valid
