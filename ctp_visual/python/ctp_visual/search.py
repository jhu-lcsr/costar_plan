from __future__ import print_function

import numpy as np

from costar_task_plan.abstract import *
from costar_models.conditional_image import ConditionalImage

class VisualSearchNode(object):
    '''
    Stores activity for a single node of the visual tree search.
    '''
    def __init__(self, cim, parent, action, q=1.):
        '''
        Create a node in the visual tree search.

        Parameters:
        -----------
        cim: conditional image model
        parent: parent node (none if this is root)
        action: integer embedding for action to perform
        q: expected q value
        '''
        # Neural net model holder/manager -- should be ConditionalImage for now
        self.cim = cim
        # This stores the parent node
        self.parent = parent
        # Action stores the action that was taken to get to this state
        self.action = np.array([action])
        # This stores the q-value getting to this state
        self.q = q
        self.visits = 0
        self.expanded = False
        self.h = None
        self.h0 = None
        self.value = None
        self.children = {}

    def expand(self, h0, h):
        if not self.expanded:
            self.h0 = h0
            self.h = self.cim.transform(self.h0, h, self.action)
            self.v = self.cim.value(self.h0, self.h)
            self.expanded = True
    
    def explore(self, depth=0, max_depth=5):

        if depth >= max_depth:
            return self.v

        # Compute expected transitions and value function for next actions
        p_a, done_a = self.cim.pnext(self.h0, self.h, self.action)
        q, done_qa = self.cim.q(self.h0, self.h, self.action)
        
        # Compute the next q and action
        print(p_a)
        print(q)
        print(done_a, done_qa)

        a = np.argmax(p_a, axis=1)[0]
        print("action =", a)
        if not a in self.children:
            self.children[a] = VisualSearchNode(
                    cim=self.cim,
                    parent=self,
                    action=a)
            self.children[a].expand(self.h0, self.h)
        node = self.children[a]

        print("action =", a, 
              "q(parent, a) =", q[0,a],
              "value =", node.v,
              "depth =", depth, "/", max_depth)
        v = node.explore(depth+1, max_depth)
        print(" -- ", v)

        # Return total value to parent (success probability)
        return self.v * v

    def makeRoot(self, I0):
        assert(self.parent is None)
        self.h = self.cim.encode(I0)
        self.h0 = self.h
        self.v = self.cim.value(self.h, self.h)
        self.prev_a = np.array([self.cim.null_option])
        self.expanded = True

class VisualSearch(object):
    '''
    Hold the tree and perform a short visual tree search.
    '''

    def __init__(self, model, *args, **kwargs):
        '''
        Create the conditional image model.
        '''
        self.cim = model
        if not isinstance(self.cim, ConditionalImage):
            raise RuntimeError('model type not supported: ' +
                    str(type(self.cim)))

    def __call__(self, I, iter=10, depth=5):
        '''
        Take in current world observation.
        Create a search tree:
         - generate nex
        '''
        self.root = VisualSearchNode(
                self.cim,
                parent=None,
                action=self.cim.null_option, 
                q=1.)
        self.root.makeRoot(I)
        for i in range(iter):
            self.root.explore(depth=0, max_depth=depth)
        


