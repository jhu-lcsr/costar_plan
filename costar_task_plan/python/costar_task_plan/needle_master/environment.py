# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:30:52 2015

@author: Chris Paxton
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
try:
  import sympy
except ImportError, e:
  print "NeedleMaster requires sympy."
  print "Try to install it with:"
  print "\tpip install sympy"
  raise e


def SafeLoadLine(name, handle):
    l = handle.readline()[:-1].split(': ')
    assert(l[0] == name)

    return l[1].split(',')


def ArrayToTuples(array):
    return zip(array[:, 0], array[:, 1])


class Environment:

    def __init__(self, filename=None):

        self.height = 0
        self.width = 0
        self.ngates = 0
        self.gates = []
        self.surfaces = []

        if not filename is None:
            print 'Loading environment from "%s"...' % (filename)
            handle = open(filename, 'r')
            self.Load(handle)
            handle.close()

    def Draw(self, gamecolor=True):
        axes = plt.gca()
        plt.ylim(self.height)
        plt.xlim(self.width)
        for surface in self.surfaces:
            surface.Draw()
        for gate in self.gates:
            gate.Draw()

    def InGate(self, demo):
        for gate in self.gates:
            print gate.Contains(demo)
        return False

    '''
    Load an environment file.
    '''

    def Load(self, handle):

        D = SafeLoadLine('Dimensions', handle)
        self.height = int(D[1])
        self.width = int(D[0])
        print " - width=%d, height=%d" % (self.width, self.height)

        D = SafeLoadLine('Gates', handle)
        self.ngates = int(D[0])
        print " - num gates=%d" % (self.ngates)

        gate_id = 0
        for i in range(self.ngates):
            gate = Gate(self.width, self.height, gate_id)
            gate.Load(handle)
            self.gates.append(gate)

        D = SafeLoadLine('Surfaces', handle)
        self.nsurfaces = int(D[0])
        print " - num surfaces=%d" % (self.nsurfaces)

        for i in range(self.nsurfaces):
            s = Surface(self.width, self.height)
            s.Load(handle)
            self.surfaces.append(s)


class Gate:

    def __init__(self, env_width, env_height, gate_id):
        self.x = 0
        self.y = 0
        self.w = 0
        self.top = np.zeros((4, 2))
        self.bottom = np.zeros((4, 2))
        self.corners = np.zeros((4, 2))
        self.width = 0
        self.height = 0
        self.gate_id = gate_id

        self.box = None
        self.bottom_box = None
        self.top_box = None

        self.env_width = env_width
        self.env_height = env_height

    def Contains(self, demo):
        # print demo.s.shape
        # print [x for x in demo.s]
        return [self.box.encloses_point(sympy.Point(x[:2])) for x in demo.s]  # , self.box.distance(sympy.Point(x[:2]))] for x in demo.s]

    def Contains(self, state):
        return self.box.encloses_point(state.vec[:2])

    def Features(self, demo):
        return False

    def Draw(self, gamecolor=True):
        c1 = [251. / 255, 216. / 255, 114. / 255]
        c2 = [255. / 255, 50. / 255, 12. / 255]
        c3 = [255. / 255, 12. / 255, 150. / 255]
        ce = [0, 0, 0]

        if not gamecolor:
            c1 = [0.95, 0.95, 0.95]
            c2 = [0.75, 0.75, 0.75]
            c3 = [0.75, 0.75, 0.75]
            ce = [0.66, 0.66, 0.66]

        axes = plt.gca()
        axes.add_patch(Polygon(ArrayToTuples(self.corners), color=c1))
        axes.add_patch(Polygon(ArrayToTuples(self.top), color=c2))
        axes.add_patch(Polygon(ArrayToTuples(self.bottom), color=c3))

    '''
    Load Gate from file at the current position.
    '''

    def Load(self, handle):

        pos = SafeLoadLine('GatePos', handle)
        cornersx = SafeLoadLine('GateX', handle)
        cornersy = SafeLoadLine('GateY', handle)
        topx = SafeLoadLine('TopX', handle)
        topy = SafeLoadLine('TopY', handle)
        bottomx = SafeLoadLine('BottomX', handle)
        bottomy = SafeLoadLine('BottomY', handle)

        self.x = self.env_width * float(pos[0])
        self.y = self.env_height * float(pos[1])
        self.w = float(pos[2])

        self.top[:, 0] = [float(x) for x in topx]
        self.top[:, 1] = [float(y) for y in topy]
        self.bottom[:, 0] = [float(x) for x in bottomx]
        self.bottom[:, 1] = [float(y) for y in bottomy]
        self.corners[:, 0] = [float(x) for x in cornersx]
        self.corners[:, 1] = [float(y) for y in cornersy]

        # apply corrections to make sure the gates are oriented right
        self.w *= -1
        if self.w < 0:
            self.w = self.w + (np.pi * 2)
        if self.w > np.pi:
            self.w -= np.pi
            self.top = np.squeeze(self.top[np.ix_([2, 3, 0, 1]), :2])
            self.bottom = np.squeeze(self.bottom[np.ix_([2, 3, 0, 1]), :2])
            self.corners = np.squeeze(self.corners[np.ix_([2, 3, 0, 1]), :2])

        self.w -= np.pi / 2

        avgtopy = np.mean(self.top[:, 1])
        avgbottomy = np.mean(self.bottom[:, 1])

        # flip top and bottom if necessary
        if avgtopy < avgbottomy:
            tmp = self.top
            self.top = self.bottom
            self.bottom = tmp

        # compute gate height and width

        # compute other things like polygon
        p1, p2, p3, p4 = [x[:2] for x in self.corners]
        self.box = sympy.Polygon(p1, p2, p3, p4)
        p1, p2, p3, p4 = [x[:2] for x in self.top]
        self.top_box = sympy.Polygon(p1, p2, p3, p4)
        p1, p2, p3, p4 = [x[:2] for x in self.bottom]
        self.bottom_box = sympy.Polygon(p1, p2, p3, p4)


class Surface:

    def __init__(self, env_width, env_height):
        self.deep = False
        self.corners = None
        self.color = [0., 0., 0.]

        self.env_width = env_width
        self.env_height = env_height

        self.poly = None

    def Draw(self):
        axes = plt.gca()
        axes.add_patch(Polygon(ArrayToTuples(self.corners), color=self.color))

    '''
    Load surface from file at the current position
    '''

    def Load(self, handle):
        isdeep = SafeLoadLine('IsDeepTissue', handle)

        sx = [float(x) for x in SafeLoadLine('SurfaceX', handle)]
        sy = [float(x) for x in SafeLoadLine('SurfaceY', handle)]
        self.corners = np.array([sx, sy]).transpose()
        self.corners[:, 1] = self.env_height - self.corners[:, 1]

        self.deep = (isdeep[0] == 'true')

        if not self.deep:
            self.color = [232. / 255, 146. / 255, 124. / 255]
        else:
            self.color = [207. / 255, 69. / 255, 32. / 255]

        self.poly = sympy.Polygon(*[x[:2] for x in self.corners])
