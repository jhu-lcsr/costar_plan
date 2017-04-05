# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from config import TOM_LEFT_CONFIG, TOM_RIGHT_CONFIG

from tum_ics_msgs.msg import VisualInfo
from geometry_msgs.msg import PoseArray, Pose
from tf_conversions import posemath as pm

from costar_task_plan.datasets import TomDataset

from costar_task_plan.robotics.core import CostarWorld
from costar_task_plan.robotics.core import DemoReward


# This is a simple world for the TOM task.
# In this task, we pick up an orange and move it to either the trash or to a
# bin. 
class TomWorld(CostarWorld):

  def __init__(self, data_root='', fake=True, load_dataset=False, *args, **kwargs):
    super(TomWorld,self).__init__(None,
        namespace='/tom',
        robot_config=[TOM_RIGHT_CONFIG, TOM_LEFT_CONFIG],
        fake=fake,
        *args, **kwargs)

    self.oranges = []

    # Remove this logic in the future. This is where we load the data set,
    # annd then use this data to create and save a bunch of DMPs corresponding
    # to the different actions we might want to take.
    if load_dataset:
      self.dataset = TomDataset()
      self.dataset.load(root_filepath=data_root)

      self.addTrajectories("move",
          self.dataset.move_trajs,
          self.dataset.move_data,
          ['time','squeeze_area'])
      self.addTrajectories("pickup",
          self.dataset.pickup_trajs,
          self.dataset.pickup_data,
          ['time', 'orange'])
      self.addTrajectories("test",
          self.dataset.test_trajs,
          self.dataset.test_data,
          ['time', 'squeeze_area'])
      self.addTrajectories("box",
          self.dataset.box,
          self.dataset.box_data,
          ['time', 'box'])
      self.addTrajectories("trash",
          self.dataset.trash,
          self.dataset.trash_data,
          ['time', 'trash'])

      self.ref_data = self.dataset.move_data + \
                         self.dataset.pickup_data + \
                         self.dataset.test_data + \
                         self.dataset.box_data + \
                         self.dataset.trash_data

      # Call the learning after we've loaded our data
      self.fitTrajectories()
      
      # update the feature function based on known object frames
      self.makeFeatureFunction()

  # This is used for putting data in the right form for learning
  def _preprocessData(self,data):
    for traj in data:
      orange_pose = None
      # find first non-None orange
      for world in traj:
        if world['orange'] is not None:
          orange_pose = world['orange']
          break
      for world in traj:
        world['orange'] = orange_pose

  # Get visualization information as a vector of poses for whatever object we
  # are currently manipulating.
  def _dataToPose(self,data):
    msg = PoseArray()
    for traj in data:
      for world in traj:
        if world['orange'] is not None:
          msg.poses.append(pm.toMsg(world['orange']))
        msg.poses.append(pm.toMsg(world['box']))
        msg.poses.append(pm.toMsg(world['trash']))
        msg.poses.append(pm.toMsg(world['squeeze_area']))
    return msg

  def make_task_plan(self):
    args = self.getArgs()

