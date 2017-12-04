from __future__ import print_function

import argparse
import sys

from costar_task_plan.simulation import GetSimulationParser

def GetLaunchOptions():
    '''
    These are the files that actually set up the environment
    '''
    return ["ur5","husky"]

def GetExperimentOptions():
    '''
    Each of these needs to be handled separately later on
    '''
    return ["magnetic_assembly",
            "stack",
            "tables",
            "navigation"]

def _assemblyCases():
    cases = ["double","training","finished_assembly"]
    for i in range(1,11):
        cases.append("double%d"%i)
    for i in range(1,8):
        cases.append("training%d"%i)
    for i in range(1,3):
        cases.append("assembly%d"%i)
    return cases

def ParseGazeboArgs():
    parser = GetSimulationParser()
    parser.add_argument('--launch',
                        help="ROS launch file to start Gazebo simulation",
                        default="ur5",
                        choices=GetLaunchOptions())
    parser.add_argument("--experiment",
                        help="Experiment file that configures task",
                        default="magnetic_assembly",
                        choices=GetExperimentOptions())
    parser.add_argument("--case",
                        help="Case for magnetic assembly experiment",
                        default="assembly1",
                        choices=_assemblyCases())
    parser.add_argument("--gzclient",
                        help="Bring up the gazebo client",
                        action="store_true")
    return vars(parser.parse_args())


