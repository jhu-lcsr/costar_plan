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
            self.value = None

        def expand(self, h0, h):
            self.h = self.cim.transform(h0, h, self.action)
            self.v = self.cim.value(self.h)
            self.expanded = True

        def makeRoot(self, I0):
            assert(self.parent is None)
            self.h = self.cim.encode(h)
            self.v = self.cim.value(self.h)
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
        

    def _expand(self, node):
        pass

