#!/usr/bin/env python

'''
Use this script to launch the whole simulation, not the various launch files.
'''

from gazebo_msgs.srv import SetModelConfiguration

import rospy
import signal
import subprocess
import sys
import time

class CostarSimulation(object):

    model_name = "robot"
    joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    def __init__(self):
        self.procs = []

    def sleep(self, t=1.0):
        time.sleep(t)

    def reset(self):
        rospy.wait_for_service("gazebo/set_model_configuration")
        configure = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        configure(model_name=self.model_name,
                joint_names=self.joint_names,
                joint_positions=self.joint_positions)

    def run(self):

        # ---------------------------------------------------------------------
        # Start the roscore
        roscore = subprocess.Popen(['roscore'])
        self.procs.append(roscore)
        self.sleep(1.)

        # We create a node handle so we can interact with ROS services.
        rospy.init_node('simulation_manager_node')
        
        # ---------------------------------------------------------------------
        # Start gazebo
        gazebo = subprocess.Popen(['roslaunch', 'costar_simulation', 'ur5.launch'])
        self.procs.append(gazebo)
        self.sleep(5.)

        # ---------------------------------------------------------------------
        # Reset the simulation. This puts the robot into its initial state, and
        # is also responsible for creating and updating positions of any objects.
        self.reset()
        self.sleep(1.)
        self.reset()

        # ---------------------------------------------------------------------
        # Start controllers
        roscontrol = subprocess.Popen(['roslaunch', 'costar_simulation', 'controllers.launch'])
        self.procs.append(roscontrol)

    def shutdown(self):
        for proc in reversed(self.procs):
            proc.terminate()
        self.sleep()
        try:
            for proc in reversed(self.procs):
                proc.kill()
        except Exception, e:
            pass

    def shutdownAndExitHandler(self, *args, **kwargs):
        print('You pressed Ctrl+C! Shutting down all processes.')
        self.shutdown()
        sys.exit(0)

if __name__ == '__main__':
    sim = CostarSimulation()
    signal.signal(signal.SIGINT, sim.shutdownAndExitHandler)
    try:
        sim.run()
    except Exception, e:
        print e
        sim.shutdown()
        sys.exit(-1)
    signal.pause()

