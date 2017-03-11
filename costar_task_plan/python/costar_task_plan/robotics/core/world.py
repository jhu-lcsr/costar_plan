# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from frame import *
from detected_object import *
from actor import *
from dynamics import *
from js_listener import JointStateListener

from costar_task_plan.abstract import *

import logging
from os.path import join
from geometry_msgs.msg import PoseArray
import tf

# use this for now -- because it's simple and works pretty well
import gmr

from ..config import DEFAULT_MODEL_CONFIG
from ..config import DEFAULT_ROBOT_CONFIG

LOGGER = logging.getLogger(__name__)

'''
This version of the world listens to objects as TF frames.
'''
class CostarWorld(AbstractWorld):
  def __init__(self, reward,
      namespace='/costar',
      fake=True,
      robot_config=None,
      cmd_parser=None,
      *args, **kwargs):
    super(CostarWorld,self).__init__(reward, *args, **kwargs)
    self.objects = {}
    self.object_classes = {}
    self.trajectories = {}
    self.trajectory_data = {}
    self.traj_pubs = {}
    self.traj_data_pubs = {}
    self.fake = fake
    self.predicates = []
    self.models = {}
    self.cmd_parser = cmd_parser

    if robot_config is None:
      robot_config = DEFAULT_ROBOT_CONFIG

    # set up actors and things
    self.namespace = robot_config['namespace']
    self.base_link = robot_config['base_link']
    self.js_listener = JointStateListener(robot_config)
    if robot_config['q0'] is not None:
      s0 = CostarState(self, q=robot_config['q0'])
    else:
      rospy.sleep(1.0)
      if self.js_listener.q0 is not None:
        s0 = CostarState(self, q=self.js_listener.q0)
      else:
        s0 = CostarState(self, q=np.zeros((robot_config['dof'],)))

    self.tf_pub = tf.TransformBroadcaster()

    self.addActor(CostarActor(robot_config, state=s0, dynamics=self.getT(robot_config)))

  '''
  Add an object to the list of tracked objects.
  '''
  def addObject(self, name, obj_class, obj):

    # Make sure this was a unique object
    if obj in self.objects:
      raise RuntimeError('Duplicate object inserted!')

    # Add the object data
    self.objects[name] = obj

    # Update object class membership
    if obj_class not in self.object_classes:
      self.object_classes[obj_class] = [name]
    else:
      self.object_classes[obj_class].append(name)

  '''
  Empty the list of objects.
  '''
  def clearObjects(self):
    self.objects = {}
    self.object_classes = {}
  
  def addTrajectories(self, name, trajectories, data):
    self.trajectories[name] = trajectories
    self.trajectory_data[data] = data
    if not name in self.traj_pubs:
      self.traj_pubs[name] = rospy.Publisher(
          join(self.namespace,"trajectories",name),
          PoseArray,
          queue_size=1000)
      self.traj_data_pubs[name] = rospy.Publisher(
          join(self.namespace,"trajectory_data",name),
          PoseArray,
          queue_size=1000)

  def parse(self, cmd):
    if self.cmd_parser is None:
      raise RuntimeError('No command parser provided.')

  def getT(self,robot_config,*args,**kwargs):
    if self.fake:
      return SimulatedDynamics(robot_config)
    else:
      return SubscriberDynamics(self.js_listener)
  
  '''
  Publish all training trajectories for visualization
  '''
  def hook(self):
    # publish trajectory demonstrations
    for name, trajs in self.trajectories.items():
      msg = PoseArray()
      msg.header.frame_id = self.base_link
      for traj in trajs:
        for t, pose, _, _, _ in traj:
          msg.poses.append(pose)
      self.traj_pubs[name].publish(msg)

    for name, data in self.trajectory_data.items():
      msg = self._dataToPose(data)
      msg.header.frame_id = self.base_link
      self.traj_data_pubs[name].publish(msg)

    # publish object frames
    for name, frame in self.objects.items():
      (trans, rot) = frame.tf_frame
      self.tf_pub.sendTransform(trans, rot,
              rospy.Time.now(),
              frame.tf_name,
              self.base_link,)

  def _dataToPose(self,data):
    return PoseArray()

  def makeFeatureFunction(self):
    pass

  def makeRewardFunction(self, name):
    if name in self.models.keys():
        model = self.models[name]
        self.reward = DemoReward(gmm=model)
    else:
        LOGGER.warning('model "%s" does not exist'%name)

  def zeroAction(self):
    q = np.zeros((self.actors[0].dof,))
    return CostarAction(dq=q)

  def fitTrajectories(self):
    self.models = {}
    for name, trajs in self.trajectories.items():
        data = []

  '''
  Gets the list of possible argument assigments for use in generating the final
  task plan object.
  '''
  def getArgs(self):
    return self.object_classes

