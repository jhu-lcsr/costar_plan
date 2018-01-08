
'''
This function contains some helpful functions for different purposes.
'''

from std_srvs.srv import Empty as EmptySrv

import rospy

def GetDetectObjectsService(segmenter_service='/costar_perception/segmenter'):
    '''
    Get a service that will update object positions
    '''
    rospy.wait_for_service(segmenter_service)
    return rospy.ServiceProxy(segmenter_service, EmptySrv)
