
# This defines a simple class that 
class AbstractGripperStatusListener(object):

    def callback(self, msg):
        raise NotImplementedError('gripper status listener must consume a ROS"
                " message.')

    def getStatus(self):
        raise NotImplementedError('gripper status listener must return a" 
        " status message for consumption by the world.')
