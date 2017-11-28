#!/usr/bin/env python

'''
Tools to collect a single data set for the robot learning task. We assume data
collection works via ros.
'''

# Import everything from the "costar workshop assistant" definition for CTP
# robotics tasks
from costar_task_plan.robotics.workshop import *

<<<<<<< HEAD
def main(args):
    '''
    Start up listeners for data collection, and run the main loop.
    '''
    pass

if __name__ == '__main__':
    args = ParseWorkshopArgs()
    main(args)
=======
def main(hz,**kwargs):
    '''
    Start up listeners for data collection, and run the main loop.
    '''
    rospy.init_node("ctp_data_collector")
    manager = ListenerManager()
    rate = rospy.Rate(hz)
    try:
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    args = ParseWorkshopArgs()
    main(**args)
>>>>>>> a8c68be115af8788d19bcd7aec6c74eee5f06225
