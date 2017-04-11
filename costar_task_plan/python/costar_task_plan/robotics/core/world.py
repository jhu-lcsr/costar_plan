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

      *args, **kwargs):
    super(CostarWorld,self).__init__(reward, *args, **kwargs)

    # This is the set of trajectory data we will use for learning later on.
    self.trajectories = {}
    
    # This is the set of object information we will use for learning later on
    # It tells us which objects/features each individual skill should depend on
    # and is used when extracting a set of features.
    self.objs = {}

    # This is extra data -- such as world state observations -- that is
    # associated with our training trajectories.
    self.trajectory_data = {}

    # This is where we actually store all the information about our learned
    # skills.
    self.models = {}

    # ------------------------- VISUALIZATION TOOLS ---------------------------
    # These are publishers for pose arrays that help us visualize data and
    # learned actions.
    self.traj_pubs = {}
    self.traj_data_pubs = {}
    self.skill_pubs = {}
    self.tf_listener = None

    # Other things
    self.fake = fake
    self.predicates = []
    self.namespace = namespace

    # This is the current state of all non-robot objects in the world --
    # which is to say, for now, it's just a dictionary of frames by TF frame
    # identifier.
    self.observation = {}

    if robot_config is None:
      raise RuntimeError('Must provide a robot config!')

    # -------------------------- ROBOT INFORMATION ----------------------------
    # Base link, end effector, etc. for easy reference
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

    # The base link for the scene as a whole
    self.base_link = self.actors[0].base_link

    self.lfd = LfD(self)
  
  # [LEARNING HELPER FUNCTION ONLY]
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

  # Execute a sequence of nodes. We use the policy and condition from each
  # node, together with a set of subscriber dynamics, to update the world.
  def execute(self, path, actor_id=0):
      # start publishers

      # loop over the path
      for node in path:
          if node.action is None:
              rospy.logwarn('Done execution.')
              break
          policy = node.action.policy
          condition = node.action.condition
          actor = self.actors[actor__id].state

          # check condition -- same loop as in the tree search.
          while not condition(self, self.actors[actor_id].state, ):
              #cmd = policy.evaluate(self, 
              pass

      # Done.

  # Create the set of dynamics used for this particular option/option distribution.
  def getT(self,robot_config,*args,**kwargs):
    if self.fake:
      return SimulatedDynamics()
    else:
      return SubscriberDynamics(self.js_listeners[robot_config['name']])
  
  # Hook is called after the world updates each actor according to its policy.
  # It has a few responsibilities:
  # 1) publish all training trajectories for visualization
  # 2) publish the current command/state associated with each actor too the sim.
  def visualize(self):
    # Publish trajectory demonstrations for easy comparison to the existing
    # stuff.
    for name, trajs in self.trajectories.items():
      msg = PoseArray()
      msg.header.frame_id = self.base_link
      for traj in trajs:
        for _, pose, _, _ in traj:
          msg.poses.append(pose)
      self.traj_pubs[name].publish(msg)

    # Publish trajectory data. These are the observed world states associated
    # with each of the various trajectories we are interested in.
    for name, data in self.trajectory_data.items():
      msg = self._dataToPose(data)
      msg.header.frame_id = self.base_link
      self.traj_data_pubs[name].publish(msg)

    # Publish actor states
    for actor in self.actors:
      msg = JointState(
          name=actor.joints,
          position=actor.state.q,
          velocity=actor.state.dq)
  def debugLfD(self, args):
    self.lfd.debug(self, args)

  # Look up what the world should look like, based on the provided arguments.
  # "objs" should be a list of all possible objects that we might want to
  # aggregate. They'll be saved in the "obs" dictionary. It's the role of the
  # various options/policies to choose and use them intelligently.
  def updateObservation(self, objs):
    self.observation = {}
    if self.tf_listener is None:
      self.tf_listener = tf.TransformListener()

    try:
      for obj in objs:
        (trans, rot) = self.tf_listener.lookupTransform(self.base_link, obj, rospy.Time(0.))
        self.observation[obj] = pm.fromTf((trans, rot))

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      pass
      


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

