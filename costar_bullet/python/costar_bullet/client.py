
from config import *

import pybullet as pb
import rospy
import subprocess
import time

'''
Wrapper for talking to a single robot.
'''
class CostarBulletSimulation(object):
    def __init__(self, robot, task, gui=False, name="simulation", *args, **kwargs):
        self.gui = gui
        self.robot = robot
        self.task = task

        # managed list of processes
        self.procs = []

        # boot up ROS and open a connection to the simulation server
        self._start_ros(name)
        self.open()

    def _start_ros(self, name):
        '''
        Simple function to boot up a ROS core and make sure that we connect to
        it. The goal is to manage internal ROS stuff via this script to provide
        an easy and repeatable interface when running experiments.
        '''
        self._start_process(['roscore'], 1.)
        started = False
        tries = 0
        while not started:
            try:
                rospy.init_node(name)
                started = True
            except Exception, e:
                pass
            finally:
                tries += 1
                time.sleep(0.1)
            if tries > 1000:
                raise RuntimeError('Could not connect to ROS core!')

    def _start_process(self, cmd, time_to_wait):
        proc = subprocess.Popen(cmd)
        self.procs.append(proc)
        time.sleep(time_to_wait)

    def open(self):
        '''
        Decide how we will create our interface to the underlying simulation.
        We can create a GUI connection or something else.
        '''
        if self.gui:
            connect_type = pb.GUI
        else:
            connect_type = pb.DIRECT
        self.client = pb.connect(connect_type)
        pb.setGravity(*GRAVITY)

    def close(self):
        '''
        Close connection to bullet sim.
        '''
        pb.disconnect()

    def step(self):
        '''
        We most likely do not need this interface, but it provides an extra
        layer of abstraction for us to shield against changes in the underlying
        pybullet API.
        '''
        pb.stepSimulation()

    def __delete__(self):
        '''
        Call the close function if this is ever garbage collected.
        '''
        self.close()
