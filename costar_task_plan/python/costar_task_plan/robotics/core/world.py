# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from frame import *
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
    self.trajectories = {}
    self.traj_pubs = {}
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
  def addObject(self, name, tf_frame, obj_class):
    self.objects[name] = Frame(name,
            obj_class,
            tf_frame,
            namespace=self.namespace)
  
  def addTrajectories(self, name, trajectories):
    self.trajectories[name] = trajectories
    if not name in self.traj_pubs:
      self.traj_pubs[name] = rospy.Publisher(
          join(self.namespace,"trajectories",name),
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

    # publish object frames
    for name, frame in self.objects.items():
      (trans, rot) = frame.tf_frame
      self.tf_pub.sendTransform(trans, rot,
              rospy.Time.now(),
              frame.tf_name,
              self.base_link,)

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

