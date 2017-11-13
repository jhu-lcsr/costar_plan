#!/usr/bin/env python

'''
Tools to collect a single data set for the robot learning task. We assume data
collection works via ros.
'''

# Import everything from the "costar workshop assistant" definition for CTP
# robotics tasks
from costar_task_plan.robotics.workshop import *

def main(args):
    '''
    Start up listeners for data collection, and run the main loop.
    '''
    pass

if __name__ == '__main__':
    args = ParseWorkshopArgs()
    main(args)
