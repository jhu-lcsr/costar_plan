#!/usr/bin/env python

'''
Use this script to launch the whole simulation, not the various launch files.
'''

from gazebo_msgs.srv import SetModelConfiguration
from std_srvs.srv import Empty as EmptySrv

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
    experiment = "assembly1"

    def __init__(self):
        self.procs = []
        self.reset_srv = None

    def sleep(self, t=1.0):
        time.sleep(t)

    def reset(self):
        self.pause()
        rospy.wait_for_service("gazebo/set_model_configuration")
        configure = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        configure(model_name=self.model_name,
                joint_names=self.joint_names,
                joint_positions=self.joint_positions)
        res = subprocess.call([
            "roslaunch",
            "costar_simulation",
            "magnetic_assembly.launch",
            "experiment:=%s"%self.experiment])
        self.resume()

    def pause(self):
        pass

    def resume(self):
        pass

    def reset_srv_cb(self, msg):
        self.reset()

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

        # ---------------------------------------------------------------------
        # Start reset service
        self.reset_srv = rospy.Service("reset_simulation", EmptySrv, self.reset_srv_cb)

    def shutdown(self):

        # Send terminate signal to all managed processes.
        for proc in reversed(self.procs):
            proc.terminate()

        # Wait and kill everything.
        self.sleep()
        try:
            for proc in reversed(self.procs):
                proc.kill()
        except Exception, e:
            pass

        # For some reason gazebo does not shut down nicely.
        self.sleep()
        subprocess.call(["pkill","gzserver"])

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

