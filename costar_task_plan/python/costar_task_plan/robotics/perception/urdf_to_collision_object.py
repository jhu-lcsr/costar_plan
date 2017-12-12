
# URDF parser to load visual and collision elements
from urdf_parser_py.urdf import URDF

from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse

import pyassimp
import PyKDL as kdl
import rospy
import tf
import tf_conversions.posemath as pm
import urdf_parser_py

def _getUrdf(name, rosparam):
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

def _make_mesh(self, c, scale = (1, 1, 1)):
    '''
    This was taken from moveit commander and slightly modified.
    '''
    filename = c.geometry.filename
    if pyassimp is False:
        raise RuntimeError("Pyassimp needs patch https://launchpadlibrarian.net/319496602/patchPyassim.txt")
    scene = pyassimp.load(filename)
    if not scene.meshes or len(scene.meshes) == 0:
        raise MoveItCommanderException("There are no meshes in the file")
    if len(scene.meshes[0].faces) == 0:
        raise MoveItCommanderException("There are no faces in the mesh")
    
    mesh = Mesh()
    first_face = scene.meshes[0].faces[0]
    if hasattr(first_face, '__len__'):
        for face in scene.meshes[0].faces:
            if len(face) == 3:
                triangle = MeshTriangle()
                triangle.vertex_indices = [face[0], face[1], face[2]]
                mesh.triangles.append(triangle)
    elif hasattr(first_face, 'indices'):
        for face in scene.meshes[0].faces:
            if len(face.indices) == 3:
                triangle = MeshTriangle()
                triangle.vertex_indices = [face.indices[0],
                                           face.indices[1],
                                           face.indices[2]]
                mesh.triangles.append(triangle)
    else:
        raise RuntimeError("Unable to build triangles from mesh due to mesh object structure")
    for vertex in scene.meshes[0].vertices:
        point = Point()
        point.x = vertex[0]*scale[0]
        point.y = vertex[1]*scale[1]
        point.z = vertex[2]*scale[2]
        mesh.vertices.append(point)
    pyassimp.release(scene)
    return mesh

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
            # Only update the geometry if we actually need to add the
            # object to the collision scene.
            # check type of each collision tag.
            if isinstance(c.geometry, urdf_parser_py.urdf.Box):
                primitive = True
                if co.operation == CollisionObject.ADD:
                    size = c.geometry.size
                    element = SolidPrimitive()
                    element.type = SolidPrimitive.BOX
                    element.dimensions = list(c.geometry.size)
                    co.primitives.append(element)
            elif isinstance(c.geometry, urdf_parser_py.urdf.Mesh):
                primitive = False
                if co.operation == CollisionObject.ADD:
                    scale = (1,1,1)
                    if c.geometry.scale is not None:
                        scale = c.scale
                    element = _loadMesh(c.geometry, scale)
                    co.meshes.append(element)
            else:
                raise NotImplementedError("we do not currently support geometry of type %s"%(str(type(c.geometry))))

            pose = kdl.Frame(
                    kdl.Rotation.RPY(*c.origin.rpy),
                    kdl.Vector(*c.origin.xyz))
            pose = link_pose * pose
            if primitive:
                co.primitive_poses.append(pm.toMsg(pose))
            else:
                # was a mesh
                co.mesh_poses.append(pm.toMsg(pose))

    return co

class CollisionObjectManager(object):
    '''
    Creates and aggregates information from multiple URDF objects and publishes
    to a planning scene based on TF positions of these objects.
    '''

    def __init__(self, root="/world", listener=None, max_dt=1.):
        self.objs = {}
        self.urdfs = {}
        self.frames = {}
        if listener is None:
            self.listener = tf.TransformListener()
        else:
            self.listener = listener
        self.root = root
        self.max_dt = max_dt

        rospy.wait_for_service('/get_planning_scene')

        self.co_pub = rospy.Publisher('collision_object',
                CollisionObject,
                queue_size=1000)

    def addUrdf(self, name, rosparam, tf_frame=None):
        urdf = _getUrdf(name, rosparam)
        self.objs[name] = None
        self.urdfs[name] = urdf
        if tf_frame is None:
            tf_frame = name
        self.frames[name] = tf_frame

    def tick(self):

        #TODO: figure out why this is bad
        #if not self.listener.frameExists(self.root):
        #    rospy.logerr("missing root: %s"%self.root)
        #    return
        self.t = rospy.Time.now()

        for name, urdf in self.urdfs.items():
            if self.objs[name] == None:
                operation = CollisionObject.ADD
            else:
                operation = CollisionObject.MOVE
            if not self.listener.frameExists(name):
                #rospy.logwarn("Frame %s does not exist"%name)
                continue
            try:
                t = self.listener.getLatestCommonTime(self.root, name)
            except tf.Exception as e:
                rospy.logerr(str(e))
                continue
            dt = (self.t - t).to_sec(
            if dt > self.max_dt:
                rospy.logwarn("object %s has not been observed in the last %f seconds"%(name, dt))
                continue
            pose = self.listener.lookupTransform(self.root, name, t)
            pose = pm.fromTf(pose)
            co = _getCollisionObject(name, urdf, pose, operation)
            co.header.frame_id = self.root
            self.objs[name] = co
            self.co_pub.publish(co)


