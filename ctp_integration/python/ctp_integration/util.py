
'''
This function contains some helpful functions for different purposes.
'''

from std_srvs.srv import Empty as EmptySrv
from costar_robot_msgs.srv import SmartMove
from costar_robot_msgs.srv import ServoToJointState
from costar_robot_msgs.srv import ServoToJointStateRequest
from costar_robot_msgs.srv import ServoToPose, SetServoMode

import rospy

def GetDetectObjectsService(srv='/costar_perception/segmenter'):
    '''
    Get a service that will update object positions

    Parameters:
    ----------
    srv: service, defaults to the correct name for costar
    '''
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetCloseGripperService(srv="/costar/gripper/close"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetCloseGripperService(srv="/costar/gripper/open"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetSmartGraspService(srv="/costar/SmartGrasp"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, SmartMove)

def GetSmartPlaceService(srv="/costar/SmartPlace"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, SmartMove)

def GetSmartReleaseService():
    srv = "/costar/SmartRelease"
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, SmartMove)

def GetServoToJointStateService(srv="/costar/ServoToJointState"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, ServoToJointState)

def GetPlanToPoseService():
    srv = "/costar/PlanToPose"
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, ServoToPose)

def GetPlanToHomeService():
    srv = "/costar/PlanToHome"
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, ServoToPose)

def GetOpenGripperService():
    srv = "/costar/gripper/open"
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetServoModeService():
    srv = "/costar/SetServoMode"
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, SetServoMode)

def MakeServoToJointStateRequest(position):
    req = ServoToJointStateRequest()
    req.target.position = position
    return req
