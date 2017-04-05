
import numpy as np
from tf_conversions import posemath as pm

# for outputting things to ROS
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from dmp_utils import RequestDMP, PlanDMP

# Model an instance of a skill as a cartesian DMP. We use this to create all
# of the different skills we may need.
class CartesianSkillInstance(object):

  # Needs:
  # - a vector of end effector poses
  # - a vector of world state observations (dictionaries)
  # - a kinematics model
  # Assume that end effector and worlds are in the same coordinate system,
  # which is supposed to be the base link.
  def __init__(self, ee_frames, worlds, kinematics, config, objs=[], dt=0.1, visualize=False):
    self.config = config
    self.ee_frames = ee_frames
    self.worlds = worlds
    self.kinematics = kinematics
    self.dt = dt
    self.objs = [obj for obj in objs if obj not in ['time', 'gripper']]
    self._fit()

  # call to create the dmp based on this observation
  def _fit(self):
    k_gain = self.config['dmp_k']
    d_gain = self.config['dmp_d']
    num_basis = self.config['dmp_basis']

    if len(self.objs) > 1:
      raise RuntimeError('CartesianSkillInstance does not handle multiple object goals!')
    elif len(self.objs) is 0:
      # goal
      pass
      goal_frame = [pm.fromMatrix(np.eye(4))] * len(self.worlds)
    else:
      goal_frame = [world[self.objs[0]] for world in self.worlds]

    u = np.zeros((len(goal_frame),6))
    last_rpy = None
    for i, (ee,goal) in enumerate(zip(self.ee_frames, goal_frame)):
        pose = goal.Inverse() * ee
        u[i,0] = pose.p[0]
        u[i,1] = pose.p[1]
        u[i,2] = pose.p[2]

        # make sure all our motions are nice and continuous -- or strange things will happen
        adj_rpy = [0,0,0]
        rpy = pose.M.GetRPY()
        if last_rpy is not None:
            for j, (lvar, var) in enumerate(zip(last_rpy, rpy)):
                if lvar < 0 and var > lvar + np.pi:
                    adj_rpy[j] = var - 2*np.pi
                elif lvar > 0 and var < lvar - np.pi:
                    adj_rpy[j] = var + 2*np.pi
                else:
                    adj_rpy[j] = var

        u[i,3] = rpy[0]
        u[i,4] = rpy[1]
        u[i,5] = rpy[2]

        # Sanity check!
        if last_rpy is not None:
            for lvar, var in zip(last_rpy, adj_rpy):
                if abs(lvar - var) > np.pi:
                    raise RuntimeError('big jump when computing angle! %f, %f'%(lvar, var))

        last_rpy = adj_rpy

    resp = RequestDMP(u,self.dt,k_gain,d_gain,num_basis)

    self.dmp_list = resp.dmp_list
    self.tau = resp.tau

  # Given a world state and a robot state, generate a trajectory. This will
  # create both the joint state
  def generate(self, world, state):
    Fx0 = self.kinematics.forward(state.q)
    x0 = [Fx0.p[0], Fx0.p[1], Fx0.p[2],]
    x0_dot = [0.,0.,0.,]
    goal_thresh = [1e-1, 1e-1, 1e-1]
    tau = 1.0
    integrate_iter = 5
    dt = 0.1
    resp = PlanDMP(x0, x0_dot, 0., goal, goal_thresh, tau, dt, integrate_iter)
    if visualize:
      msg = PoseArray()
    pass

