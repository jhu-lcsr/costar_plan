
import numpy as np
from tf_conversions import posemath as pm

# for outputting things to ROS
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from dmp_utils import RequestDMP, PlanDMP

# Model an instance of a skill as a cartesian DMP. We use this to create all
# of the different skills we may need.
class CartesianSkillInstance(object):

  # Needs:
  # - a vector of end effector poses
  # - a vector of world state observations (dictionaries)
  # - a kinematics model
  # Assume that end effector and worlds are in the same coordinate system,
  # which is supposed to be the base link.
  def __init__(self, ee_frames, worlds, kinematics, config, objs=[], visualize=False):
    self.config = config
    self.ee_frames = ee_frames
    self.worlds = worlds
    self.kinematics = kinematics
    self.objs = [obj for obj in objs if obj not in ['time', 'gripper']]
    self._fit()

  # call to create the dmp based on this observation
  def _fit(self):
    k_gain = self.config['dmp_k']
    d_gain = self.config['dmp_d']
    n_basis = self.config['dmp_basis']

    if len(self.objs) > 1:
      raise RuntimeError('CartesianSkillInstance does not handle multiple object goals!')
    elif len(self.objs) is 0:
      # goal
      pass
      goal_frame = [pm.fromMatrix(np.eye(4))] * len(self.worlds)
    else:
      goal_frame = [world[self.objs[0]] for world in self.worlds]

    resp = RequestDMP(u,dt,k_gain,d_gain,num_basis_functions)
    pass

  # Given a world state and a robot state, generate a trajectory. THis will
  # create both the joint state
  def generate(self, world, state):
    if visualize:
      msg = PoseArray()
    pass

