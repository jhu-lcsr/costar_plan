
'''
This function contains some helpful functions for different purposes.
'''

from std_srvs.srv import Empty as EmptySrv
from costar_robot_msgs.srv import SmartMove
from costar_robot_msgs.srv import ServoToJointState
from costar_robot_msgs.srv import ServoToJointStateRequest
from costar_robot_msgs.srv import ForwardKinematicsRequest
from costar_robot_msgs.srv import ForwardKinematics
from costar_robot_msgs.srv import ServoToPose
from costar_robot_msgs.srv import SetServoMode

import rospy

def GetService(name, config_class):
    service = None
    while service is None:
        rospy.wait_for_service(name)
        service = rospy.ServiceProxy(name, config_class)
    return service

def GetDetectObjectsService(srv='/costar_perception/segmenter'):
    '''
    Get a service that will update object positions

    Parameters:
    ----------
    srv: service, defaults to the correct name for costar
    '''
    return GetService(srv, EmptySrv)

def GetCloseGripperService():
    return GetService("/costar/gripper/close", EmptySrv)

def GetSmartGraspService(srv="/costar/SmartGrasp"):
    return GetService(srv, SmartMove)

def GetSmartPlaceService(srv="/costar/SmartPlace"):
    return GetService(srv, SmartMove)

def GetSmartReleaseService():
    return GetService("/costar/SmartRelease", SmartMove)

def GetServoToJointStateService():
    return GetService("/costar/ServoToJointState", ServoToJointState)

def GetPlanToJointStateService():
    return GetService("/costar/PlanToJointState", ServoToJointState)

def GetPlanToPoseService():
    return GetService("/costar/PlanToPose", ServoToPose)

def GetForwardKinematicsService():
    return GetService("/costar/ForwardKinematics", ForwardKinematics)

def GetPlanToHomeService():
    return GetService("/costar/PlanToHome", ServoToPose)

def GetOpenGripperService():
    return GetService("/costar/gripper/open", EmptySrv)

def GetServoModeService():
    return GetService("/costar/SetServoMode", SetServoMode)

def MakeServoToJointStateRequest(position):
    req = None
    while req is None:
        req = ServoToJointStateRequest()
    req.target.position = position
    req.vel = 1.0
    req.accel = 0.75
    return req

def MakeForwardKinematicsRequest(position):
    req = None
    while req is None:
        req = ForwardKinematicsRequest()
    req.joint_state.position = position
    return req
