
# URDF parser to load visual and collision elements
from urdf_parser_py import URDF

from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse

import pyassimp
import rospy

def GetUrdfCollisionObject(name, rosparam=None, urdf=None):
    '''
    This function loads a collision object specified as an URDF for integration
    with perception and MoveIt.
    '''
    if rospy.is_shutdown():
        raise RuntimeError('URDF must be loaded from the parameter server.')
    elif rosparam is None and urdf is None:
        raise RuntimeError('no model found!')

    if urdf is None:
        urdf = URDF.from_parameter_server(rosparam)
        add = True
    else:
        add = False

    co = CollisionObject()
    co.id = name
    if add:
        co.operation = CollisionObject.ADD
    else:
        co.operation = CollisionObject.MOVE

    for l in urdf.links:
        if co.operation == CollisionObject.ADD:
            for c in l.collisions:
                # check type
                if isinstance(c, urdf_parser_py.urdf.Box):
                    size = c.size
                    element = SolidPrimitive
                    element.type = SolidPrimitive.BOX
                    element.dimensions = list(c.geometry.size)
                    co.primitives.append(element)
                    co.primitive_poses.append(pose)

    return co, urdf

class CollisionObjectManager(object):
    def __init__(self):
        self.objs = {}
        self.urdfs = {}
        self.frames = {}

    def addUrdf(self, name, rosparam, tf_frame=None):
        co, urdf = GetUrdfCollisionObject(name, rosparam, None)
        self.objs[name] = co
        self.urdfs[name] = urdf
        if tf_frame is None:
            tf_frame = name
        self.frames[name] = tf_frame

    def tick(self):
        for name, urdf in self.urdfs.items():
            co, _ = GetUrdfCollisionObject(name, rosparam=None, urdf=urdf)


