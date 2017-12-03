#!/usr/bin/env python

'''
Tools to collect a single data set for the robot learning task. We assume data
collection works via ros.
'''

import rospy

# Import everything from the "costar workshop assistant" definition for CTP
# robotics tasks
from costar_task_plan.robotics.workshop import *

def main(hz,**kwargs):
    '''
    Start up listeners for data collection, and run the main loop.
    '''
    rospy.init_node("ctp_data_collector")
    manager = ListenerManager(**kwargs)
    rate = rospy.Rate(hz)
    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.RosInterruptException as e:
        pass

if __name__ == '__main__':
    args = ParseWorkshopArgs()
    main(**args)

