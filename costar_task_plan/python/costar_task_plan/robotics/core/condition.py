from costar_task_plan.abstract import AbstractCondition

from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

import rospy

class ValidStateCondition(AbstractCondition):
    '''
    Check to see if a particular state is valid
    '''

    def __init__(self):
        self.srv = rospy.ServiceProxy('check_state_validity', GetStateValidity)

    def __call__(self, world, state, actor=None, prev_state=None):
        if actor is None:
            actor = world.actors[0]
        js = JointState(
                name=actor.joints,
                position=state.q,
                )
        rs = RobotState(joint_state=js)
        print "----"
        print rs
        res = self.srv(robot_state=rs)
        print res
        return res.valid
    
