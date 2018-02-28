from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from costar_task_plan.abstract import *
from costar_models.conditional_image import ConditionalImage

class VisualSearchNode(object):
    '''
    Stores activity for a single node of the visual tree search.
    '''
    def __init__(self, cim, parent, action):
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
        self.visits = 0
        self.expanded = False
        self.h = None
        self.h0 = None
        self.value = None
        self.children = {}

        self.done = 0.
        self.q = None
        self.child_max = np.zeros((self.cim.num_options,))
        self.child_value = np.zeros((self.cim.num_options,))
        self.child_visits = np.ones((self.cim.num_options,))
        self.terminal = False

    def _expand(self, h0, h):
        if not self.expanded:
            self.h0 = h0
            self.h = self.cim.transform(self.h0, h, self.action)
            self.v = self.cim.value(self.h0, self.h)
            self.expanded = True

            # Compute probabilities and next probabilities
            p_a, done_a = self.cim.pnext(self.h0, self.h, self.action)
            q, done_qa = self.cim.q(self.h0, self.h, self.action)
            self.q = q[0]
            self.done = done_qa[0]
            self.child_value = self.q
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
        self.child_visits[action] += 1
        if self.child_max[action] < value:
            self.child_max[action] = value
        #self.child_value = (10 * self.q / self.child_visits) + self.child_max
        #self.child_value /= np.sum(self.child_value)
    
    def explore(self, depth=0, max_depth=5, i=0, num_iter=0, draw=False):

        if depth >= max_depth or self.terminal:
            return self.v

        # Compute the next q and action
        print ("--------------", self.action, depth)
        #print(self.q, self.done, self.child_value)

        #a = np.argmax(p_a, axis=1)[0]
        a = np.random.choice(range(self.cim.num_options),p=self.child_value)
        print("action =", a)
        if not a in self.children:
            self.children[a] = VisualSearchNode(
                    cim=self.cim,
                    parent=self,
                    action=a)
            self.children[a]._expand(self.h0, self.h)

        if draw and num_iter > 0:
            idx = (max_depth * i) + (depth+1)
            plt.subplot(num_iter, max_depth, idx)
            plt.axis('off')
            plt.tight_layout()
            plt.title("%d %d: a=%d d=%d"%(i, idx, a, (depth)))
            plt.imshow(self.cim.decode(self.children[a].h)[0], interpolation='nearest')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

        node = self.children[a]

        print("action =", a, 
              "q(parent, a) =", self.q[a],
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
        self.prev_a = np.array([self.cim.null_option])
        self.expanded = True
        self.done = 1.

        p_a, done_a = self.cim.pnext(self.h0, self.h, self.action)
        q, done_qa = self.cim.q(self.h0, self.h, self.action)
        self.q = q[0]
        self.child_value = self.q

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

    def __call__(self, I, iter=10, depth=5, draw=False):
        '''
        Take in current world observation.
        Create a search tree:
         - generate nex
        '''
        self.root = VisualSearchNode(
                self.cim,
                parent=None,
                action=self.cim.null_option,)
        self.root.makeRoot(I)
        for i in range(iter):
            self.root.explore(depth=0, max_depth=depth, i=i, num_iter=iter,
                    draw=draw)
        


