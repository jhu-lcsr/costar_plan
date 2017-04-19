import rospy
from costar_task_plan.mcts import AbstractExecute

from sensor_msgs.msg import JointState

# This sends a single point (and associated joints) off to the remote robot.
class TomPointExecute(AbstractExecute):
    def __init__(self, joints, dt, namespace="",):
        self.publish = rospy.Publisher("joint_states_cmd",JointState,queue_size=1000).publish
        self.joints = joints
        self.dt = dt

    def __call__(self, cmd):
        msg = JointState(name=self.joints,
                         position=cmd.q)
        self.publish(msg)
        
