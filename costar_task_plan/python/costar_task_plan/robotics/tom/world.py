# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from config import TOM_LEFT_CONFIG, TOM_RIGHT_CONFIG

from tum_ics_msgs.msg import VisualInfo
from geometry_msgs.msg import PoseArray, Pose

from costar_task_plan.datasets import TomDataset

from costar_task_plan.robotics.core import CostarWorld
from costar_task_plan.robotics.core import DemoReward
from costar_task_plan.robotics.core import DmpOption

# Set up the "pick" action that we want to performm
def __pick_args():
  return {
    "constructor": TomDmpOption,
    "args": ["orange","kinematics"],
    "remap": {"orange": "goal_frame"},
      }

# Instantiate the whole task model based on our data. We must make sure to
# provide the lfd object containing models, etc., or we will not properly
# create all of the different DMP models.
def MakeTomTaskModel(lfd):
  task = Task()


# This is a simple world for the TOM task.
# In this task, we pick up an orange and move it to either the trash or to a
# bin. 
class TomWorld(CostarWorld):

  # These are the preset positions for the various TOM objects. These are 
  # reference frames used for computing features. These are the ones
  # associated with the main TOM dataset.
  box = (0.67051013617,
         -0.5828498549,
         -0.280936861547)
  squeeze_area = (0.542672622341,
                  0.013907504104,
                  -0.466499112972)
  trash = (0.29702347941,
           0.0110837137159,
           -0.41238342306)

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
          self.dataset.move_data,)
      self.addTrajectories("pickup",
          self.dataset.pickup_trajs,
          self.dataset.pickup_data,)
      self.addTrajectories("test",
          self.dataset.test_trajs,
          self.dataset.test_data,)
      self.addTrajectories("box",
          self.dataset.box,
          self.dataset.box_data,)
      self.addTrajectories("trash",
          self.dataset.trash,
          self.dataset.trash_data,)

      self.ref_data = self.dataset.move_data + \
                         self.dataset.pickup_data + \
                         self.dataset.test_data + \
                         self.dataset.box_data + \
                         self.dataset.trash_data

      self.fitTrajectories()
      
      # update the feature function based on known object frames
      self.makeFeatureFunction()

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
          msg.poses.append(world['orange'])
        msg.poses.append(world['box'])
        msg.poses.append(world['trash'])
        msg.poses.append(world['squeeze_area'])
    return msg

  def vision_cb(self, msg):

    self.clearObjects()
    self.addObject("box", "box", TomObject(pos=self.box))
    self.addObject("squeeze_area", "squeeze_area", TomObject(pos=self.squeeze_area))
    self.addObject("trash", "trash", TomObject(pos=self.trash))

    self.oranges = []

    for obj in msg.objData:
      if obj.objType == "Orange":
        # add it to our list of detected objects
        self.addObject("orange",
            "orange%d"%count,
            TomObject(msg=obj))

  def make_task_plan(self):
    args = self.getArgs()

# Represent the TOM orange as an object in the world.
class TomObject(object):
  def __init__(self, msg=None, pos=None):
    if msg is not None:
      self.x = msg.position.x
      self.y = msg.position.y
      self.z = msg.position.z
    elif pos is not None:
      self.x = pos[0]
      self.y = pos[1]
      self.z = pos[2]
    else:
      raise RuntimeError('Must provide either a message or a tuple!')

