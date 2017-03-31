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

def __pick_args():
  return {
    "constructor": TomDmpOption,
    "args": ["orange"],
    "remap": {"orange": "goal_frame"},
      }

def MakeTomTaskModel():
  task = Task()


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

    # These are the preset positions for the various TOM objects. These are 
    # reference frames used for computing features. These are the ones
    # associated with the main TOM dataset.
    box = (0.67051013617,
           -0.5828498549,
           -0.280936861547), Pose
    squeeze_area = (0.542672622341,
                    0.013907504104,
                    -0.466499112972)
    trash = (0.29702347941,
             0.0110837137159,
             -0.41238342306)

    # Rotation frame for all of these is pointing down at the table.
    rot = (0, 0, 0, 0)

    # Remove this logic in the future. This is where we load the data set,
    # annd then use this data to create and save a bunch of DMPs corresponding
    # to the different actions we might want to take.
    if load_dataset:
      self.dataset = TomDataset()
      self.dataset.load(root_filepath=data_root)

      self.addTrajectories("move",
          self.dataset.move_trajs,
          self.dataset.move_oranges,)
      self.addTrajectories("pickup",
          self.dataset.pickup_trajs,
          self.dataset.pickup_oranges,)
      self.addTrajectories("test",
          self.dataset.test_trajs,
          self.dataset.test_oranges,)
      self.addTrajectories("box",
          self.dataset.box,
          self.dataset.box_oranges,)
      self.addTrajectories("trash",
          self.dataset.trash,
          self.dataset.trash_oranges,)

      self.ref_oranges = self.dataset.move_oranges + \
                         self.dataset.pickup_oranges + \
                         self.dataset.test_oranges + \
                         self.dataset.box_oranges + \
                         self.dataset.trash_oranges

      self.box = ['box']
      self.trash = ['trash']
      self.squeeze_area = ['squeeze_area']
      self.oranges = []

      self.fitTrajectories()
      
      # update the feature function based on known object frames
      self.makeFeatureFunction()

  def _dataToPose(self,data):
    msg = PoseArray()
    for data_traj in data:
      for orange in data_traj:
        if orange is not None:
          msg.poses.append(Pose(position=(orange.position)))
    return msg

  def vision_cb(self, msg):

    self.clearObjects()
    self.addObject("box", "box", TomObject(pos=box))
    self.addObject("squeeze_area", "squeeze_area", TomObject(pos=squeeze_area))
    self.addObject("trash", "trash", TomObject(pos=trash))

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

