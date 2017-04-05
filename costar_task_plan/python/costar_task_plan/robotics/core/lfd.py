
import numpy as np
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf_conversions import posemath as pm
from urdf_parser_py.urdf import URDF

# Include representations for either (a) the generic skill or (b) an instance
# of a skill represented as a goal-directed set of motion primitives.
from costar_task_plan.robotics.representation import RobotFeatures
from costar_task_plan.robotics.representation import CartesianSkillInstance
from costar_task_plan.robotics.representation import GMM

# Compute features for trajectories, objects.
# This all must take a world.
class LfD(object):

  def __init__(self, world):
    self.world = world

    config = world.actors[0].config
    base_link = config['base_link']
    end_link = config['end_link']

    if 'robot_description_param' in config:
      self.robot = URDF.from_parameter_server(config['robot_description_param'])
    else:
      self.robot = URDF.from_parameter_server()

    self.config = config

    # set up kinematics stuff
    self.tree = kdl_tree_from_urdf_model(self.robot)
    self.chain = self.tree.getChain(base_link, end_link)
    self.kdl_kin = KDLKinematics(self.robot, base_link, end_link)

    self.base_link = base_link
    self.end_link = end_link

    self.skill_instances = {}
    self.skill_features = {}
    self.skill_models = {}


  # Train things
  def train(self):

    for name, trajs in self.world.trajectories.items():

      data = self.world.trajectory_data[name]
      features = RobotFeatures(self.config, self.kdl_kin)
      objs = self.world.objs[name]

      self.skill_instances[name] = []

      for traj, world in zip(trajs, data):

        ts = [t for t,_,_,_ in traj]
        dt = np.mean(np.diff(ts))

        ee = [pm.fromMsg(pose) for _,pose,_,_ in traj]
        gripper = [gopen for _,_,gopen,_ in traj]

        if not len(ee) == len(gripper) and len(gripper) == len(world):
          raise RuntimeError('counting error')

        # compute features?
        f,g = features.GetFeaturesForTrajectory(ee, world[0], objs)
        instance = CartesianSkillInstance(ee,
            world,
            self.kdl_kin,
            self.config,
            dt=dt,
            objs=objs,
            visualize=True)

        self.skill_instances[name].append(instance)
        if name not in self.skill_features:
          self.skill_features[name] = f
        else:
          np.concatenate((self.skill_features[name], f), axis=0)

        print name, self.skill_features[name].shape
        self.skill_models[name] = GMM(self.config['gmm_k'], self.skill_features[name])

    return self.skill_models

  # Save models after they have been fit.
  def save(self, dir='./data/'):
    pass


  # Save models after they have been fit.
  def load(self, dir='./data/'):
    pass

