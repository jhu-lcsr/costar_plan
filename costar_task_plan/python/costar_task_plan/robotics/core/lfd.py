
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
      features = RobotFeatures(self.config, self.kdl_kin)

      for traj, world in zip(trajs, data):
        print len(traj), len(world)
        print world[0]['orange']

        break
      break
