
from tf_conversions import posemath as pm
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF

from costar_task_plan.robotics.representation import RobotFeatures

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


  # Train things
  def train(self):

    for name, trajs in self.world.trajectories.items():

      data = self.world.trajectory_data[name]
      print "==============="
      print len(trajs), len(data)
      features = RobotFeatures(self.config, self.kdl_kin)
      objs = self.world.objs[name]

      for traj, world in zip(trajs, data):

        ee = [pm.fromMsg(pose) for _,pose,_,_ in traj]
        gripper = [gopen for _,_,gopen,_ in traj]

        if not len(ee) == len(gripper) and len(gripper) == len(world):
          raise RuntimeError('counting error')

        # compute features?
        f,g = features.GetFeaturesForTrajectory(ee, world[0], objs)

        # update

      break
