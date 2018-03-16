#!/usr/bin/env python

import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import rospy
import tf_conversions.posemath as pm
import tf2_ros as tf2

from costar_task_plan.mcts import PlanExecutionManager, DefaultExecute
from costar_task_plan.mcts import OptionsExecutionManager
from costar_task_plan.robotics.core import RosTaskParser
from costar_task_plan.robotics.core import CostarWorld
from costar_task_plan.robotics.workshop import UR5_C_MODEL_CONFIG
from sensor_msgs.msg import JointState

from ctp_integration import MakeStackTask
from ctp_integration.observer import IdentityObserver, Observer
from ctp_integration.collector import DataCollector
from ctp_integration.util import GetDetectObjectsService
from ctp_integration.stack import *

def getArgs():
    '''
    Get argument parser and call it to get information from the command line.

    Parameters:
    -----------
    none

    Returns:
    --------
    args: command-line arguments
    '''
    parser = argparse.ArgumentParser(add_help=True, description="Parse rosbag into graph.")
    parser.add_argument("--fake",
                        action="store_true",
                        help="create some fake options for stuff")
    parser.add_argument("--show",
                        action="store_true",
                        help="show a graph of the compiled task")
    parser.add_argument("--plan",
                        action="store_true",
                        help="set if you want the robot to generate a task plan")
    parser.add_argument("--execute",
                        type=int,
                        help="execute this many loops",
                        default=1)
    parser.add_argument("--start",
                        type=int,
                        help="start from this index",
                        default=1)
    parser.add_argument("--iter","-i",
                        default=0,
                        type=int,
                        help="number of samples to draw")
    parser.add_argument("--mode",
                        choices=["collect","test"],
                        default="collect",
                        help="Choose which mode to run in.")
    parser.add_argument("--verbose", "-v",
                        type=int,
                        default=0,
                        help="verbosity level")
    parser.add_argument("--rate", "-r",
                        default=10,
                        type=int,
                        help="rate at which data will be collected")

    return parser.parse_args()

def fakeTaskArgs():
  '''
  Set up a simplified set of arguments. These are for the optimization loop, 
  where we expect to only have one object to grasp at a time.
  '''
  args = {
    'block': ['block_1', 'block_2'],
    'endpoint': ['r_ee_link'],
    'high_table': ['ar_marker_2'],
    'Cube_blue': ['blue1'],
    'Cube_red': ['red1'],
    'Cube_green': ['green1'],
    'Cube_yellow': ['yellow1'],
  }
  return args

def main():

    # Create node and read options from command line
    rospy.init_node("ctp_data_collection_runner")
    args = getArgs()

    # Default joints for the motion planner when it needs to go to the home
    # position - this will take us out of the way of the camera.
    try:
        q0 = rospy.get_param('/costar/robot/home')
    except KeyError as e:
        rospy.logwarn("CoSTAR home position not set, using default.")
        q0 = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    # Create the task model, world, and other tools
    rospy.loginfo("Making stack task...")
    task = MakeStackTask()
    rospy.loginfo("Making world...")
    world = CostarWorld(robot_config=UR5_C_MODEL_CONFIG)
    rospy.loginfo("Aggregating TF data...")
    tf_buffer = tf2.Buffer()
    tf_listener = tf2.TransformListener(tf_buffer)
    
    rospy.loginfo("Node started, waiting for transform data...")
    rospy.sleep(0.5) # wait to cache incoming transforms

    if args.fake:
        world.addObjects(fakeTaskArgs())
        filled_args = task.compile(fakeTaskArgs())
        observe = IdentityObserver(world, task)
    else:
        objects = GetDetectObjectsService()
        observe = Observer(world=world,
                task=task,
                detect_srv=objects,
                topic="/costar_sp_segmenter/detected_object_list",
                tf_listener=tf_buffer)

    # print out task info
    if args.verbose > 0:
        print(task.nodeSummary())
        print(task.children['ROOT()'])

    stack_task = GetStackManager()
    collector = DataCollector(
            task=stack_task,
            data_root="~/.costar/data",
            rate=args.rate,
            data_type="h5f",
            robot_config=UR5_C_MODEL_CONFIG,
            camera_frame="camera_link",
            tf_listener=tf_buffer)
    rate = rospy.Rate(args.rate)
    home = GetHome()
    move_to_pose = GetMoveToPose()
    open_gripper = GetOpenGripperService()
    close_gripper = GetCloseGripperService()
    update = GetUpdate(observe, collector) # uses collector because it listens for js
    stack_task.setUpdate(update) # set fn to call after actions

    # How we verify the objet
    def verify(object_name):
        '''
        Simple verify functor. This is designed to work if we have one object
        of each color, and is not guaranteed to work otherwise.

        Parameters:
        -----------
        object_name: name of the object being manipulated
        '''
        try:
            t = rospy.Time(0)
            pose = collector.tf_listener.lookup_transform(collector.base_link, object_name, t)
        except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
            rospy.logwarn("Failed lookup: %s to %s"%(collector.base_link, object_name))
            return False
        print(">>>", object_name, pose)
        return pose.transform.translation.z > 0.10

    start = max(0, args.start-1)
    i = start
    while i < args.execute:
        home()
        rospy.sleep(0.1) # Make sure nothing weird happens with timing
        idx = i + 1
        print("Executing trial %d..."%(idx))
        _, world = observe()
        # NOTE: not using CTP task execution framework right now
        # It's just overkill
        #names, options = task.sampleSequence()
        #plan = OptionsExecutionManager(options)

        # Reset the task manager
        reward = 0.
        stack_task.reset()

        poses = []
        # Update the plan and the collector in synchrony.
        while not rospy.is_shutdown():

            # Note: this will be "dummied out" for most of 
            done = stack_task.tick()
            if not collector.update(stack_task.current, ):
                raise RuntimeError('could not handle data collection')
            rate.sleep()

            if (collector.prev_action is not None and
                not collector.prev_action == collector.action and
                "place" in collector.prev_action):
                poses.append(collector.current_ee_pose)
                reward = 0.

            if done:
                rospy.logwarn("DONE WITH: " + str(stack_task.ok))
                if stack_task.ok:
                    # Increase count
                    i += 1

                    # We should actually check results here
                    home(); observe()
                    rospy.sleep(0.5)
                    if verify(collector.prev_object):
                        reward = 1.
                    else:
                        reward = 0.
                    rospy.loginfo("reward = " + str(reward))
                break

        if stack_task.ok:
            collector.save(idx, reward)
            print("------------------------------------------------------------")
            print("Finished one round of data collection. Please reset the test")
            print("environment before continuing.")
            print("")
            print("Example number:", idx, "/", args.execute)
            print("Success:", reward)
            print("")
        else:
            print("------------------------------------------------------------")
            print("Bad data collection round. Manually reset.")
            collector.reset()

        try:
            input("Press Enter to continue...")
        except SyntaxError as e:
                pass

        # Undo the stacking
        for drop_pose in reversed(poses):
            grasp_pose = copy.deepcopy(drop_pose)
            grasp_pose.p[2] -= 0.05 # should be smart release backoff distance
            x = 0.43 + (0.3 * np.random.random())
            y = -0.08 - (0.22 * np.random.random())
            z = 0.3
            pose_random = kdl.Frame(drop_pose.M,
                    kdl.Vector(x,y,z))
            rospy.logwarn(str(drop_pose))
            move_to_pose(drop_pose)
            rospy.logwarn(str(grasp_pose))
            move_to_pose(grasp_pose)
            close_gripper()
            rospy.logwarn(str(pose_random))
            move_to_pose(pose_random)
            open_gripper()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        pass
