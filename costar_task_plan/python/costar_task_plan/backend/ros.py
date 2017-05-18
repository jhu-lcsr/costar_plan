import rospy
import rospkg

class RosBackend(AbstractBackend):
    '''
    Use ROS tools to find packages, communicate between nodes, etc.
    '''

    def __init__(self, node_name, *args, **kwargs):
        super(RosBackend).__init__(self, *args, **kwargs)
        rospy.init_node(node_name)

    def findPackage(self, name):
        raise RuntimeError('backend functionality not implemented')
