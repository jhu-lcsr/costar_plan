
# for outputting things to ROS
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointStates

from dmp_utils import *

# Model an instance of a skill as a cartesian DMP. We use this to create all
# of the different skills we may need.
def CartesianSkillInstanceModel(object):

  # Needs:
  # - a vector of end effector poses
  # - a vector of world state observations (dictionaries)
  # - a kinematics model
  # Assume that end effector and worlds are in the same coordinate system,
  # which is supposed to be the base link.
  def __init__(self, ee_frames, worlds, kinematics, visualize=False):
    self.end_link = config['end_link']
    self.base_link = config['base_link']

  # call to create the dmp based on this observation
  def _fit(self):
    pass

  # Given a world state and a robot state, generate a trajectory. THis will
  # create both the joint state
  def generate(self, world, state):
    if visualize:
      msg = PoseArray()
    pass

