
import pybullet as pb

from config import *

'''
Wrapper for talking to a single robot.
'''
class Client(object):
    def __init__(self, gui=False, *args, **args):
        self.gui = gui
        self.open()

    def open(self):
        if self.gui:
            connect_type = p.GUI
        else:
            connect_type = p.DIRECT
        self.client = pb.connect(connect_type)
        pb.setGravity(*GRAVITY)

    def close(self):
        pb.disconnect()

    def __delete__(self):
        self.close()
