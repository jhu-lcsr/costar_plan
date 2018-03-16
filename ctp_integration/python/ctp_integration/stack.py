from __future__ import print_function

import PyKDL as kdl
import rospy
import sys
import tf_conversions.posemath as pm

from geometry_msgs.msg import Pose
from costar_robot_msgs.srv import SmartMoveRequest
from costar_robot_msgs.srv import ServoToJointStateRequest
from costar_robot_msgs.srv import ServoToPoseRequest
from costar_robot_msgs.msg import Constraint
from std_srvs.srv import EmptyRequest
from std_srvs.srv import Empty as EmptySrv
from costar_task_plan.abstract.task import *

from .stack_manager import *

colors = ["red", "blue", "yellow", "green"]

def GetPoses():
    '''
    All poses have been recorded relative to /base_link. If the robot moves
    they may no longer work.

    This creates the poses necessary to make certain colorful patterns on the
    bottom of the white tray.
    '''
    pose1_left = kdl.Frame(
            kdl.Rotation.Quaternion(),
            kdl.Vector(0.493, -0.202, 0.216))
    pose2_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.610, 0.318, -0.549, 0.474),
            kdl.Vector(0.450, -0.304, 0.216))
    pose3_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.603, 0.320, -0.528, 0.505),
            kdl.Vector(0.557, -0.336, 0.198))
    pose4_left = kdl.Frame(
            kdl.Rotation.Quaternion(0.627, 0.320, -0.518, 0.486),
            kdl.Vector(0.594, -0.228, 0.205))
    pose1_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.650, 0.300, -0.451, 0.533),
            kdl.Vector(0.492, 0.013, 0.214))
    pose2_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.645, 0.304, -0.467, 0.523),
            kdl.Vector(0.480, -0.089, 0.210))
    pose3_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.657, 0.283, -0.472, 0.514),
            kdl.Vector(0.569, -0.110, 0.198))
    pose4_right = kdl.Frame(
            kdl.Rotation.Quaternion(0.638, 0.330, -0.421, 0.553),
            kdl.Vector(0.596, -0.014, 0.203))
    pose_home = kdl.Frame(
            kdl.Rotation.Quaternion(0.711, -0.143, -0.078, 0.684),
            kdl.Vector(0.174, -0.157, 0.682))
    poses = {"home": pose_home,
             "pose1_left": pose1_left,
             "pose2_left": pose2_left,
             "pose3_left": pose3_left,
             "pose4_left": pose4_left,
             "pose1_right": pose1_right,
             "pose2_right": pose2_right,
             "pose3_right": pose3_right,
             "pose4_right": pose4_right,}
    return poses

def GetGraspPose():
    # Grasp from the top, centered (roughly)
    pose = kdl.Frame(
            kdl.Rotation.Quaternion(1.,0.,0.,0.),
            kdl.Vector(-0.22, -0.02, -0.01))
    return pose

def GetStackPose():
    # Grasp from the top, centered (roughly)
    pose = kdl.Frame(
            kdl.Rotation.Quaternion(1.,0.,0.,0.),
            kdl.Vector(-0.22, -0.02, -0.01))
    pose = pose * kdl.Frame(kdl.Vector(-0.052,0.,0.))
    return pose

def GetTowerPoses():
    pose1 = kdl.Frame(
            kdl.Rotation.Quaternion(0.580, 0.415, -0.532, 0.456),
            kdl.Vector(0.533, -0.202, 0.234))
    
    poses = {"tower1": pose1,}
    return poses

def _makeSmartPlaceRequest(poses, name):
    '''
    Helper function for making the place call
    '''
    req = SmartMoveRequest()
    req.pose = pm.toMsg(poses[name])
    req.name = name
    req.obj_class = "place"
    req.backoff = 0.05
    return req

def GetMoveToPose():
    move_srv = GetPlanToPoseService()
    servo_mode = GetServoModeService()
    def move(pose):
        req = ServoToPoseRequest()
        req.target = pm.toMsg(pose)
        move_srv(req)
    return move

def GetHome():
    js_home = GetPlanToHomeService()
    pose_home = kdl.Frame(
            kdl.Rotation.Quaternion(0.711, -0.143, -0.078, 0.684),
            kdl.Vector(0.174, -0.157, 0.682))
    req = ServoToPoseRequest()
    req.target = pm.toMsg(pose_home)
    open_gripper = GetOpenGripperService()
    move = GetPlanToPoseService()
    servo_mode = GetServoModeService()
    def home():
        servo_mode("servo")
        open_gripper()
        res1 = js_home(ServoToPoseRequest())
        if "failure" in res1.ack.lower():
            rospy.logerr(res1.ack)
            sys.exit(-1)
        res2 = move(req)
        if "failure" in res2.ack.lower():
            rospy.logerr("move failed:" + str(res2.ack))
            sys.exit(-1)
    return home

def GetUpdate(observe, collector):
    '''
    Parameters:
    -----------
    observe: callable functor that will call object detection
    collector: wrapper class aggregating robot info including joint state
    '''
    go_to_js = GetPlanToJointStateService()
    pose_home = kdl.Frame(
            kdl.Rotation.Quaternion(0.711, -0.143, -0.078, 0.684),
            kdl.Vector(0.174, -0.157, 0.682))
    req = ServoToPoseRequest()
    req.target = pm.toMsg(pose_home)
    move = GetPlanToPoseService()
    servo_mode = GetServoModeService()
    def update():
        q0 = collector.q
        servo_mode("servo")
        res = move(req)
        if "failure" in res.ack.lower():
            rospy.logerr(res.ack)
            sys.exit(-1)
            return False
        observe()
        res2 = go_to_js(MakeServoToJointStateRequest(q0))
        if "failure" in res2.ack.lower():
            rospy.logerr(res2.ack)
            return False
            sys.exit(-1)
        else:
            return True
    return update

def GetStackManager():
    sm = StackManager()
    grasp = GetSmartGraspService()
    release = GetSmartReleaseService()

    for color in colors:
        name = "1:grab_%s"%color
        req = _makeSmartGraspRequest(color)
        sm.addRequest(None, name, grasp, req)

        for color2 in colors:
            if color2 == color:
                continue
            else:
                name2 = "2%s:place_%s_on_%s"%(color2,color,color2)
                req2 = _makeSmartReleaseRequest(color2)
                sm.addRequest(name, name2, release, req2)

                for color3 in colors:
                    if color3 in [color, color2]:
                        continue
                    else:
                        name3 = "3%s%s:grab_%s"%(color,color2,color3)
                        req3 = _makeSmartGraspRequest(color3)
                        sm.addRequest(name2, name3, grasp, req3)
                        name4 = "4%s%s:place_%s_on_%s%s"%(color,color2,color3,color2,color)
                        req4 = _makeSmartReleaseRequest(color)
                        sm.addRequest(name3, name4, release, req4)

    return sm

def _makeSmartGraspRequest(color):
    '''
    Helper function to create a grasp request via smartmove.
    '''
    req = SmartMoveRequest()
    req.pose = pm.toMsg(GetGraspPose())
    if not color in colors:
        raise RuntimeError("color %s not recognized" % color)
    req.obj_class = "%s_cube" % color
    req.name = "grasp_%s" % req.obj_class
    req.backoff = 0.05
    return req

def _makeSmartReleaseRequest(color):
    '''
    Helper function for making the place call
    '''
    constraint = Constraint(
            pose_variable=Constraint.POSE_Z,
            threshold=0.015,
            greater=True)
    req = SmartMoveRequest()
    req.pose = pm.toMsg(GetStackPose())
    if not color in colors:
        raise RuntimeError("color %s not recognized" % color)
    req.obj_class = "%s_cube" % color
    req.name = "place_on_%s" % color
    req.backoff = 0.05
    req.constraints = [constraint]
    return req

def MakeStackTask():
    '''
    Create a version of the robot task for stacking two blocks.
    '''

    # Make services
    rospy.loginfo("Waiting for SmartMove services...")
    rospy.wait_for_service("/costar/SmartPlace")
    rospy.wait_for_service("/costar/SmartGrasp")
    place = rospy.ServiceProxy("/costar/SmartPlace", SmartMove)
    grasp = rospy.ServiceProxy("/costar/SmartGrasp", SmartMove)

    # Create sub-tasks for left and right
    rospy.loginfo("Creating subtasks...")
    pickup_left = _makePickupLeft()
    pickup_right = _makePickupRight()
    place_left = _makePlaceLeft()
    place_right = _makePlaceRight()

    # Create the task: pick up any one block and put it down in a legal
    # position somewhere on the other side of the bin.
    rospy.loginfo("Creating task...")
    task = Task()
    task.add("pickup_left", None, pickup_left)
    task.add("pickup_right", None, pickup_right)
    task.add("place_left", "pickup_right", place_left)
    task.add("place_right", "pickup_left", place_right)
    task.add("DONE", ["place_right", "place_left"], {})

    return task

def _makePickupLeft():
    pickup = TaskTemplate("pickup_left", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePickupRight():
    pickup = TaskTemplate("pickup_right", None)
    pickup.add("home", None, _homeArgs())
    pickup.add("detect_objects", "home", _detectObjectsArgs())

    return {"task": pickup, "args": ["object"]}

def _makePlaceLeft():
    place = TaskTemplate("place_left", ["pickup_right"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _makePlaceRight():
    place = TaskTemplate("place_right", ["pickup_left"])
    place.add("home", None, _homeArgs())
    place.add("detect_objects", "home", _detectObjectsArgs())
    return {"task": place, "args": ["frame"]}

def _pickupLeftArgs():
    # Create args for pickup from left task
    return {
        "task": pickup_left,
        "args": ["block1"],
    }

def _pickupRightArgs():
    # And create args for pickup from right task
    return {
        "task": pickup_right,
        "args": ["block1"],
    }

def _homeArgs():
    return {}

def _detectObjectsArgs():
    return {}

def _checkBlocks1And2(block1,block2,**kwargs):
    '''
    Simple function that is passed as a callable "check" when creating the task
    execution graph. This makes sure we don't build branches that just make no
    sense -- like trying to put a blue block on top of itself.

    Parameters:
    -----------
    block1: unique block name, e.g. "red_block"
    block2: second unique block name, e.g. "blue_block"
    '''
    return not block1 == block2
