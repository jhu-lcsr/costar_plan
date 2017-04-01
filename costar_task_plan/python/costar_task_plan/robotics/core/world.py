# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from lfd import LfD
from frame import *
from detected_object import *
from actor import *
from dynamics import *
from js_listener import JointStateListener

from costar_task_plan.abstract import *
from costar_task_plan.robotics.representation import GMM

import logging
from os.path import join
from geometry_msgs.msg import PoseArray
import tf
from tf_conversions import posemath as pm

# use this for now -- because it's simple and works pretty well
import gmr

from ..config import DEFAULT_MODEL_CONFIG
from ..config import DEFAULT_ROBOT_CONFIG

LOGGER = logging.getLogger(__name__)

# This is the basic Robotics world class.
# This version of the world listens to objects as TF frames.
# POLICIES:
#  - managed policies that listen to CoSTAR proper, or something else like that
#  - specific policies that follow DMPs, etc
# At the end of every loop, we can publish all the information associated with
# each step.
class CostarWorld(AbstractWorld):
  def __init__(self, reward,
      namespace = '/costar',
      fake=True,
      robot_config=None,
      cmd_parser=None,
      *args, **kwargs):
    super(CostarWorld,self).__init__(reward, *args, **kwargs)
    self.objects = {}
    self.object_classes = {}
    self.trajectories = {}
    self.objs = {}
    self.trajectory_data = {}
    self.traj_pubs = {}
    self.traj_data_pubs = {}
    self.fake = fake
    self.predicates = []
    self.models = {}
    self.cmd_parser = cmd_parser
    self.namespace = namespace

    if robot_config is None:
      robot_config = [DEFAULT_ROBOT_CONFIG]

    # set up actors and things
    self.js_listeners = {} 
    self.tf_pub = tf.TransformBroadcaster()

    # Create and add all the robots we want in this world.
    for robot in robot_config:
      if robot['q0'] is not None:
        s0 = CostarState(self, q=robot['q0'])
      else:
        js_listener = JointStateListener(robot)
        self.js_listeners[robot['name']] = js_listener
        rospy.sleep(0.1)
        if js_listener.q0 is not None:
          s0 = CostarState(self, q=js_listener.q0)
        else:
          s0 = CostarState(self, q=np.zeros((robot['dof'],)))

      self.addActor(CostarActor(robot,
        state=s0,
        dynamics=self.getT(robot),
        policy=NullPolicy()))

    self.lfd = LfD(self)

  # Helper function to add an object to the list of tracked objects.
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

  # Empty the list of objects.
  def clearObjects(self):
    self.objects = {}
    self.object_classes = {}
  
  # Add a bunch of trajectory for use in learning.
  def addTrajectories(self, name, trajectories, data, objs):
    self.trajectories[name] = trajectories
    self.objs[name] = objs
    self._preprocessData(data)
    self.trajectory_data[name] = data
    if not name in self.traj_pubs:
      self.traj_pubs[name] = rospy.Publisher(
          join(self.namespace,"trajectories",name),
          PoseArray,
          queue_size=1000)
      self.traj_data_pubs[name] = rospy.Publisher(
          join(self.namespace,"trajectory_data",name),
          PoseArray,
          queue_size=1000)

  # Parse a command
  def parse(self, cmd):
    if self.cmd_parser is None:
      raise RuntimeError('No command parser provided.')

  # Create the set of dynamics used for this particular option/option distribution.
  def getT(self,robot_config,*args,**kwargs):
    if self.fake:
      return SimulatedDynamics(robot_config)
    else:
      return SubscriberDynamics(self.js_listeners[robot_config['name']])
  
  # Hook is called after the world updates each actor according to its policy.
  # It has a few responsibilities:
  # 1) publish all training trajectories for visualization
  # 2) publish the current command/state associated with each actor too the sim.
  def hook(self):

    # Publish trajectory demonstrations for easy comparison to the existing
    # stuff.
    for name, trajs in self.trajectories.items():
      msg = PoseArray()
      msg.header.frame_id = self.actors[0].base_link
      for traj in trajs:
        for _, pose, _, _ in traj:
          msg.poses.append(pose)
      self.traj_pubs[name].publish(msg)
    for name, data in self.trajectory_data.items():
      msg = self._dataToPose(data)
      msg.header.frame_id = self.actors[0].base_link
      self.traj_data_pubs[name].publish(msg)

    # Publish object frames
    for name, frame in self.objects.items():
      (trans, rot) = frame.tf_frame
      self.tf_pub.sendTransform(trans, rot,
              rospy.Time.now(),
              frame.tf_name,
              self.base_link,)

    # Publish actor states
    for actor in self.actors:
      msg = JointState(
          name=actor.joints,
          position=actor.state.q,
          velocity=actor.state.dq)

  # Overload this to set up data visualization; it should return a pose array.
  def _dataToPose(self,data):
    return PoseArray()

  # Process the data set
  def _preprocessData(self,data):
    pass

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

  # Create te
  def fitTrajectories(self):
    self.models = self.lfd.train()

  # Gets the list of possible argument assigments for use in generating the
  # final task plan object.
  def getArgs(self):
    return self.object_classes

