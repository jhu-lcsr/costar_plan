
import pybullet as pb

from config import *

'''
Wrapper for talking to a single robot.
'''
class BulletClient(object):
    def __init__(self, robot, task, gui=False, *args, **args):
        self.gui = gui
        self.robot = robot
        self.task = task
        self.open()

    def open(self):
        '''
        Decide how we will create our interface to the underlying simulation.
        We can create a GUI connection or something else.
        '''
        if self.gui:
            connect_type = p.GUI
        else:
            connect_type = p.DIRECT
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
