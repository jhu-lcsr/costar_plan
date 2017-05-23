import os
import rospy
from sensor_msgs.msg import JointState

from costar_task_plan.mcts import AbstractExecute

# This sends a single point (and associated joints) off to the remote robot.
class OpenLoopTomExecute(AbstractExecute):
    def __init__(self, world, actor_id, namespace="",):

        # We only really want the publish function -- and I wanted to see if
        # this would work. It does.
        self.publish = rospy.Publisher(
                os.path.join(namespace, "joint_states_cmd"),
                JointState,
                queue_size=1000).publish

        # Store which joints we are controlling.
        self.world = world
        self.joints = self.world.actors[actor_id].joints

    def __call__(self, cmd):

        if cmd.q.shape[0] == 0:
            raise RuntimeError('Passed empty joint state to execution!')

        msg = JointState(name=self.joints,
                         position=cmd.q)

        self.publish(msg)
        self.world.tick(cmd)
        
