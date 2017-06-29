'''
By Chris Paxton
Copyright (c) 2017, The Johns Hopkins University
All rights reserved.

This license is for non-commercial use only, and applies to the following
people associated with schools, universities, and non-profit research institutions

Redistribution and use in source and binary forms by the aforementioned
people and institutions, with or without modification, are permitted
provided that the following conditions are met:

* Usage is non-commercial.

* Redistribution should be to the listed entities only.

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from costar_task_plan.tools import showTask

from config import *
from features import GetFeatures
from util import GetTaskDefinition, GetRobotInterface
from world import *

import matplotlib.pyplot as plt
import numpy as np
import os
import png

from PIL import Image

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
            features="",
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
        features = GetFeatures(features)
        self.task = GetTaskDefinition(task, self.robot, features, *args, **kwargs)

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
        
    def convertToArmandGripperCmd(self, action):
        
        #TODO: fix the hard coded indices
        arm = action[0:6]
        gripper = action[5:6]
        
        print ">>>>arm " + str(len (arm))
        print ">>>>arm " + str(type(arm))
        print ">>>>arm " + str(arm)
                
        
        print ">>>>gripper " + str(len (gripper))
        print ">>>>gripper " + str(type(gripper))
        print ">>>>gripper " + str(gripper)
        
        
        return (arm,gripper)

        
    

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
            if type(action) is not tuple:
                action = self.convertToArmandGripperCmd(action)

            action = SimulationRobotAction(*action)

        # Get state, action, features, reward from update
        (ok, S0, A0, S1, F1, reward) = self.task.world.tick(action)
        
        if self.save:
            temp_data = []
            if os.path.isfile('data.npz'):
                in_data = np.load('data.npz')
                s0 = in_data['s0'].append(s0)
                A0 = in_data['A0'].append(A0)
                S1 = in_data['S1'].append(S1)
                F1 = in_data['F1'].append(F1)
                reward = in_data['reward']  
                
            np.savez('data', s0=s0, A0=A0, S1=S1, F1=F1, reward=reward)

        if self.load:
            in_data = np.load('data.npz')
            s0 = in_data['s0']
            A0 = in_data['A0']
            S1 = in_data['S1']
            F1 = in_data['F1']
            reward = in_data['reward']
            
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
                    path1 = os.path.join(self.directory,
                            "%s%04d_rgb.png"%(name, self.task.world.ticks))
                    path2 = os.path.join(self.directory,
                            "%s%04d_depth.png"%(name, self.task.world.ticks))
                    path3 = os.path.join(self.directory,
                            "%s%04d_label.png"%(name, self.task.world.ticks))


                    #img = png.fromarray(rgb, "L")
                    #temp_im_rgb = Image.fromarray(rgb, mode='L')
                    #temp_im_rgb.save(path)
                    #temp_im_rgb = cv2.imwrite(path1, rgb)

                    plt.imsave(path1, rgb)
                    plt.imsave(path2, depth)
                    plt.imsave(path3, mask)

                    #temp1 = plt.imsave(path, rgb)		    
                    #img.save(path)

            # TODO: handle other stuff

        # Return world information
        return F1, reward, not ok, {}

    def close(self):
        '''
        Close connection to bullet sim.
        '''
        pb.disconnect()

    def __delete__(self):
        '''
        Call the close function if this is ever garbage collected.
        '''
        self.close()
