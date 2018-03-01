from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from costar_task_plan.abstract import *
from costar_models.conditional_image import ConditionalImage
from costar_models.planner import GetOrderedList

class VisualSearchNode(object):
    '''
    Stores activity for a single node of the visual tree search.
    '''
    def __init__(self, task, cim, parent, action):
        '''
        Create a node in the visual tree search.

        Parameters:
        -----------
        cim: conditional image model
        parent: parent node (none if this is root)
        action: integer embedding for action to perform
        q: expected q value
        '''
        self.task = task
        # Neural net model holder/manager -- should be ConditionalImage for now
        self.cim = cim
        # This stores the parent node
        self.parent = parent
        # Action stores the action that was taken to get to this state
        self.action = np.array([action])
        # This stores the q-value getting to this state
        self.visits = 0
        self.expanded = False
        self.h = None
        self.h0 = None
        self.value = None
        self.children = {}

        self.done = 0.
        self.q = None
        self.p = None
        self.child_max = np.zeros((self.cim.num_options,))
        self.child_value = np.zeros((self.cim.num_options,))
        self.child_p = np.zeros((self.cim.num_options,))
        self.child_visits = np.ones((self.cim.num_options,))
        self.terminal = False

    def _expand(self, h0, h):
        if not self.expanded:
            self.h0 = h0
            self.h = self.cim.transform(self.h0, h, self.action)
            self.expanded = True
            self._setPQ()

            self.v = self.cim.value(self.h0, self.h)
            print (self.v, self.done)

            if self.done < 0.1:
                # Was not able to finish the motion
                self.terminal = True
                self.v *= self.done
            elif self.v < 0.1:
                # Almost definitely a failure
                self.terminal = True
        else:
            raise RuntimeError('tried to expand a node that was already'
                               'expanded!')

    def _update(self, action, value):
        '''
        Update the stored value of a particular action if it's better
        '''
        #print(" --> ", value, "from", action)
        #print(action, value, self.child_value[action], self.q[action],
        #        self.child_visits[action], self.child_max[action])
        self.child_visits[action] += 1
        if self.child_max[action] < value:
            self.child_max[action] = value
            self.best = action
        cv = self.child_value[action]
        self.child_value[action] = ((self.q[action] / self.child_visits[action])
                + (self.child_max[action]))
        print("updating (%d, %d): %f -> %f v=%f (%s)" %(self.action, action, cv,
                self.child_value[action],value,self.task.names[action]))
        self.child_p = self.child_value / np.sum(self.child_value)
    
    def explore(self, depth=0, max_depth=5, i=0, num_iter=0, draw=False):

        if depth >= max_depth or self.terminal:
            return self.v

        a = np.random.choice(range(self.cim.num_options),p=self.child_p)
        #a = np.argmax(self.child_value)
        if self.child_value[a] < 0.01:
            # print debyug info
            print("weirdly low", self.child_value[a], self.task.names[a], a)
            print(self.child_value)
            qc = GetOrderedList(self.child_value)
            pc = GetOrderedList(self.p)
            print(qc)
            print(pc)
            qq = qc[1:5]
            for qqq in qq:
                if qqq < self.cim.null_option:
                    print(self.task.names[qqq], self.q[qqq])
        if not a in self.children:
            self.children[a] = VisualSearchNode(
                    task=self.task,
                    cim=self.cim,
                    parent=self,
                    action=a)
            self.children[a]._expand(self.h0, self.h)

        if draw and num_iter > 0:
            idx = (max_depth * i) + (depth+1)
            plt.subplot(num_iter, max_depth, idx)
            plt.axis('off')
            plt.tight_layout()
            #plt.title(self.task.names[a])
            plt.imshow(self.cim.decode(self.children[a].h)[0], interpolation='nearest')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

        node = self.children[a]

        print("action =", a, self.action,
              "q(a, parent) =", self.q[a],
              "p(a | parent) =", self.p[a],
              "cv =", self.child_value[a],
              "value =", node.v,
              "depth =", depth, "/", max_depth)
        v = node.explore(depth+1, max_depth, i, num_iter, draw)
        self._update(a, v)

        # Return total value to parent (success probability)
        return self.v * v

    def makeRoot(self, I0):
        assert(self.parent is None)
        self.h = self.cim.encode(I0)
        self.h0 = self.h
        self.v = self.cim.value(self.h, self.h)
        self.action = np.array([self.cim.null_option])
        self.expanded = True
        self.done = 1.

        self._setPQ()

    def _setPQ(self):
        p_a, done_a = self.cim.pnext(self.h0, self.h, self.action)
        q, done_qa = self.cim.q(self.h0, self.h, self.action)
        self.p = p_a[0]
        self.q = q[0]
        self.q[self.action] = 0
        self.p[self.action] = 0
        self.q[self.q < 1e-2] = 0.
        self.p[self.p < 1e-2] = 0.
        self.child_value = np.copy(self.q)
        self.child_p = self.child_value / np.sum(self.child_value)
        self.done = done_qa[0]

class VisualSearch(object):
    '''
    Hold the tree and perform a short visual tree search.
    '''

    def __init__(self, task, model, *args, **kwargs):
        '''
        Create the conditional image model.
        '''
        self.task = task
        self.cim = model
        if not isinstance(self.cim, ConditionalImage):
            raise RuntimeError('model type not supported: ' +
                    str(type(self.cim)))

    def __call__(self, I, iter=10, depth=5, draw=False):
        '''
        Take in current world observation.
        Create a search tree:
         - generate nex
        '''
        self.root = VisualSearchNode(
                self.task,
                self.cim,
                parent=None,
                action=self.cim.null_option,)
        self.root.makeRoot(I)
        if draw:
            plt.figure()
        for i in range(iter):
            print("----------------- %d -------------------"%(i))
            self.root.explore(depth=0, max_depth=depth, i=i, num_iter=iter,
                    draw=draw)
        if draw:
            plt.show()
        


