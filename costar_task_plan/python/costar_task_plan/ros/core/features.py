
from costar_task_plan.abstract import AbstractFeatures

'''
Standard CoSTAR features:
  - take a list of coordinate frames
  - compute distance and relative orientation to all of these
'''
class CostarFeatures(AbstractFeatures):

  # the farthest away something can be is 3 meters
  max_distance = 3.

  def __init__(self,
      ee_frame="/end_link",
      base_frame="/base_link",
      world_frame="/world",
      frames=[]):

    self.ee_frame = ee_frame
    self.base_frame = base_frame
    self.world_frame = world_frame
    self.frames = frames
