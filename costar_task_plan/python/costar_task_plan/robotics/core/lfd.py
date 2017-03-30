
'''
Compute features for trajectories, objects.
This all must take a world.
'''
class LfD(object):

  def __init__(self, world):
    self.world = world

  # Train things
  def train(self):
    for name, traj in self.world.trajectories.items():
      data = self.world.trajectory_data[name]
  
      for obj in data:
        pass
