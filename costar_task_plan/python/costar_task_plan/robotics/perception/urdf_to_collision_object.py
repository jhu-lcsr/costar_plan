
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

def GetUrdfCollisionObject(name, rosparam):
    '''
    This function loads a collision object specified as an URDF for integration
    with perception and MoveIt.
    '''
    if rospy.is_shutdown():
        raise RuntimeError('URDF must be loaded from the parameter server.')
    elif rosparam is None:
        raise RuntimeError('no model found!')

    urdf = URDF.from_parameter_server(rosparam)

    return urdf

def _getCollisionObject(name, urdf, pose, operation):
    '''
    This function takes an urdf and a TF pose and updates the collision
    geometry associated with the current planning scene.
    '''
    co = CollisionObject()
    co.id = name
    co.operation = operation

    if len(urdf.links) > 1:
        rospy.logwarn('collison object parser does not currently support kinematic chains')

    for l in urdf.links:
        # Compute the link pose based on the origin
        if l.origin is None:
            link_pose = pose
        else:
            link_pose = pose * kdl.Frame(
                    kdl.Rotation.RPY(*l.origin.rpy),
                    kdl.Vector(*l.origin.xyz))
        for c in l.collisions:
            # check type
            if co.operation == CollisionObject.ADD:
                # Only update the geometry if we actually need to add the
                # object to the collision scene.
                if isinstance(c, urdf_parser_py.urdf.Box):
                    size = c.size
                    element = SolidPrimitive
                    element.type = SolidPrimitive.BOX
                    element.dimensions = list(c.geometry.size)
                    co.primitives.append(element)
                elif isinstance(c, urdf_parser_py.urdf.Mesh):
                    scale = (1,1,1)
                    if c.scale is not None:
                        scale = c.scale
                    element = _loadMesh(c, scale)
                    co.meshes.append(element)

            pose = kdl.Frame(
                    kdl.Rotation(*c.origin.rpy),
                    kdl.Vector(*c.origin.xyz))
            pose = link_pose * pose
            if primitive:
                co.primitive_poses.append(pose)
            else:
                # was a mesh
                co.mesh_poses.append(pose)

    return co

class CollisionObjectManager(object):
    '''
    Creates and aggregates information from multiple URDF objects and publishes
    to a planning scene based on TF positions of these objects.
    '''

    def __init__(self, root="/world", tf_listener=None):
        self.objs = {}
        self.urdfs = {}
        self.frames = {}
        if tf_listener is None:
            self.listener = tf.TransformListener()
        else:
            self.listener = tf_listener
        self.root = root

    def addUrdf(self, name, rosparam, tf_frame=None):
        urdf = _getUrdf(name, rosparam)
        self.objs[name] = None
        self.urdfs[name] = urdf
        if tf_frame is None:
            tf_frame = name
        self.frames[name] = tf_frame

    def tick(self):
        for name, urdf in self.urdfs.items():
            if self.objs[name] == None:
                operation = CollisionObject.ADD
            else:
                operation = CollisionObject.MOVE
            co = _getCollisionObject(name, urdf, pose, operation)


