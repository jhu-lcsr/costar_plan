# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from tum_ics_msgs.msg import VisualInfo

from costar_task_plan.datasets import TomDataset

from world import *
from demo_reward import *

'''
This is a simple world for the TOM task.
'''
class TomWorld(CostarWorld):

  def __init__(self, data_root='', fake=True, *args, **kwargs):
    '''
    Drop positions for the various TOM objects:

    Box:
     x: 0.67051013617
     y: -0.5828498549
     z: -0.280936861547

    Squeeze_Area:
     x: 0.542672622341
     y: 0.013907504104
     z: -0.466499112972

    Trash:
     x: 0.29702347941
     y: 0.0110837137159
     z: -0.41238342306
    '''
    
    box = (0.67051013617,
           -0.5828498549,
           -0.280936861547)
    squeeze_area = (0.542672622341,
                    0.013907504104,
                    -0.466499112972)
    trash = (0.29702347941,
             0.0110837137159,
             -0.41238342306)

    # pointing down at the table
    rot = (0, 0, 0, 0)

    robot_config = {
      'robot_description_param': "robot_description",
      'ee_link': "r_ee_link",
      'base_link': "torso_link",
      'joint_states_topic': "/joint_states",
      'dof': 6,
      'q0': None,
      'namespace': 'tom',
      'joints': ['r_shoulder_pan_joint',
        'r_shoulder_lift_joint',
        'r_elbow_joint',
        'r_wrist_1_joint',
        'r_wrist_2_joint',
        'r_wrist_3_joint']
    }

    self.dataset = TomDataset()
    self.dataset.load(root_filepath=data_root)
    super(TomWorld,self).__init__(None,
        fake=fake,
        robot_config=robot_config,
        *args, **kwargs)

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
    for orange in data:
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
'''
Represent the TOM orange as an object in the world.
'''
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

