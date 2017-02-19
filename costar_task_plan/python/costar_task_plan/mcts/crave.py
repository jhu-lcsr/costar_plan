
from abstract import AbstractScore

'''
Implement select based on continuous rave.

This is a more complicated score:
  - we need to maintain a list of all the different things that are part of
  our tree.
  - whenever the tree is updated, we have to be able to call this function.
  - luckily we can change things in the main class too...
'''
def ContinuousRaveScore(AbstractScore):

  def __init__(self, ucb1_c, alpha):
    self.ucb1_c = ucb1_c
    self.alpha = alpha
    self.next_state_idx = 0

    # store distance metrics used between the different actions
    self.state_keys = {}

    self.state_dists = []
    self.all_nodes = []

  '''
  Call is going to update everything in our internal memory.
  Procedure:
  - compute distance between child states
  - count up all states with child.action.id
  '''
  def __call__(self, parent, child):
    # add the child's value to the current list.
    child_state = child.world.actors[0].state.toArray()

    # UPDATE PROCEDURE
    if not child_state in self.state_keys:
      self.state_dists.append([])
      for i, (key, val) in enumerate(self.state_keys.items()):
        d = np.linalg.norm(child_state - key)
        self.state_dists[-1].append(d)
        self.state_dists[i].append(d)

      self.state_keys[child_state] = self.next_state_idx
      self.next_state_idx += 1

    # COMPUTE VALUE FOR ACTIONS
    action_crave = [node.avg_reward
        for node in self.all_nodes
        if node.action.id == child.action.id]

    pass
   
  '''
  Update everything in the tree
  '''
  def hook(self, root):
    pass
