from costar_task_plan import *
from costar_models import ConditionalImage

class VisualSearch(object):
    '''
    Hold the tree and perform a short visual tree search.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Create the conditional image model.
        '''
        self.cim = ConditionalImage(*args, **kwargs)

    class VisualSearchNode(object):
        '''
        Stores activity for a single node of the visual tree search.
        '''
        def __init__(self, cim, parent, action, q):
            '''
            Create a node in the visual tree search.

            Parameters:
            -----------
            cim: conditional image model
            parent: parent node (none if this is root)
            action: integer embedding for action to perform
            q: expected q value
            '''
            self.action = action
            self.visits = 0
            self.expanded = False
            self.q = q
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
        
        def explore(self, depth=0, maxdepth=5):
            # Compute expected transitions and value function for next actions
            p_a, done_a = self.cim.pnext(self.h0, self.h, self.prev_a)
            q, done_qa = self.cim.q(self.h0, self.h, self.prev_a)
            
            # Compute the next q and action
            print(p_a)
            print(q)

            a = np.argmax(p_a, axis=1)
            if not a in self.children:
                self.children[a] = VisualSearchNode(action=a)
                self.children[a].expand(self.h0, self.h)
            node = self.children[a]

            print("action =", a, 
                  "q(parent, a) =", q[0,a],
                  "value =", node.v,
                  "depth =", depth, "/", maxdepth)
            v = node.explore(depth+1, maxdepth)
            print(" -- ", V)
    
            # Return total value to parent (success probability)
            return self.v * V

        def makeRoot(self, I0):
            assert(self.parent is None)
            self.h = self.cim.encode(h)
            self.h0 = self.h
            self.v = self.cim.value(self.h)
            self.prev_a = np.array([self.cim.null_action])
            self.expanded = True

    def run(self, I):
        '''
        Take in current world observation.
        Create a search tree:
         - generate nex
        '''
        self.root = VisualSearchNode(
                self.cim,
                parent=None,
                action=self.cim.null_action, 
                q=1.)
        self.root.makeRoot(I)
        

    def _expand(self, node):
        pass

