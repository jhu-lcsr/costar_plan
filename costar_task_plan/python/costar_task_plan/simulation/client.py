
from costar_task_plan.tools import showTask

from config import *
from util import GetTaskDefinition, GetRobotInterface
from world import *

import matplotlib.pyplot as plt
import numpy as np
import os
import png
import pybullet as pb
import rospy
import subprocess
import sys
import time

class CostarBulletSimulation(object):
    '''
    Wrapper for talking to a single robot and performing a task. This brings
    things up if necessary. All this functionality could be moved into the Bullet
    gym environment.
    '''

    def __init__(self, robot, task,
            gui=False,
            ros=False,
            ros_name="simulation",
            option=None,
            plot_task=False,
            directory='./',
            save=False,
            capture=False,
            show_images=False,
            randomize_color=False,
            *args, **kwargs):
        self.gui = gui and not plot_task
        self.robot = GetRobotInterface(robot)
        self.task = GetTaskDefinition(task, self.robot, *args, **kwargs)

        # managed list of processes and other metadata
        self.procs = []
        self.ros = ros

        # saving
        self.save = save
        self.capture = capture or show_images
        self.show_images = show_images
        self.directory = directory
        self.randomize_color = randomize_color

        if ros:
            # boot up ROS and open a connection to the simulation server
            self._start_ros(ros_name)

        self.open()

        if randomize_color:
            for i in xrange(pb.getNumJoints(self.robot.handle)):
                color = np.random.random((4,))
                color[3] = 1.
                pb.changeVisualShape(self.robot.handle, i, rgbaColor=color,
                        physicsClientId=self.client)

        if plot_task:
            showTask(self.task.task)
            sys.exit(0)

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
        '''
        Simple internal function to launch a process and wait for some amount
        of time to pass. These are added to the client's list of managed
        processes so we can kill them all later.
        '''
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

        # place the robot in the world and set up the task
        self.task.setup()

    def getReward(self):
        return self.task.getReward()

    def reset(self):
        '''
        Reset the robot and task
        '''
        self.task.reset()

    def tick(self, action):
        '''
        Parse action via the robot
        '''
        #cmd = []
        #if type(action) is tuple:
        #    for term in action:
        #        cmd += term.tolist()
        #else:
        #    cmd = action.tolist()
        if not isinstance(action, SimulationRobotAction):
            action = SimulationRobotAction(*action)
        self.task.world.tick(action)

        if self.capture:
            imgs = self.task.capture()
            for name, rgb, depth, mask in imgs:
                if self.show_images:
                    plt.subplot(1,3,1)
                    plt.imshow(rgb, interpolation="none")
                    plt.subplot(1,3,2)
                    plt.imshow(depth, interpolation="none")
                    plt.subplot(1,3,3)
                    plt.imshow(mask, interpolation="none")
                    plt.pause(0.01)
                if self.save:
                    path = os.path.join(self.directory,
                            "%s%04d_rgb.png"%(name, self.task.world.ticks))
                    img = png.fromarray(rgb, "L")
                    img.save(path)

            # TODO: handle other stuff

    def close(self):
        '''
        Close connection to bullet sim.
        '''
        pb.disconnect()

    def observe(self):
        pass

    def __delete__(self):
        '''
        Call the close function if this is ever garbage collected.
        '''
        self.close()
