
from dmp.msg import DMPData
from dmp_utils import RequestDMP, PlanDMP
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from tf_conversions import posemath as pm

import numpy as np
import PyKDL as kdl
import yaml

try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

class CartesianSkillInstance(yaml.YAMLObject):
  '''
  Model an instance of a skill as a cartesian DMP. We use this to create all
  of the different skills we may need.
  '''

  yaml_tag = u'!CartesianSkillInstance'

  def __init__(self, config, params=None, objs=[], dt=0.1):
    '''
    Needs:
    - a vector of end effector poses
    - a vector of world state observations (dictionaries)
    Assume that end effector and worlds are in the same coordinate system,
    which is supposed to be the base link.
    '''
    self.config = config
    self.dt = dt
    self.objs = [obj for obj in objs if obj not in ['time', 'gripper']]
    if params is not None:
        self._fromParams(params)


  def fit(self, ee_frames, worlds):
    '''
    call to create the dmp based on this observation
    '''

    assert len(worlds) == len(ee_frames)

    k_gain = self.config['dmp_k']
    d_gain = self.config['dmp_d']
    num_basis = self.config['dmp_basis']

    if len(self.objs) > 1:
      raise RuntimeError('CartesianSkillInstance does not handle multiple object goals!')
    elif len(self.objs) is 0:
      # goal
      pass
      goal_frame = [pm.fromMatrix(np.eye(4))] * len(worlds)
    else:
      print "creating goal w.r.t. ", self.objs[0]
      goal_frame = [world[self.objs[0]] for world in worlds]

    u = np.zeros((len(goal_frame),6))
    last_rpy = None

    for i, (ee,goal) in enumerate(zip(ee_frames, goal_frame)):
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
        else:
            adj_rpy = rpy

        u[i,3] = adj_rpy[0]
        u[i,4] = adj_rpy[1]
        u[i,5] = adj_rpy[2]

        # Sanity check!
        if last_rpy is not None:
            for lvar, var in zip(last_rpy, adj_rpy):
                if abs(lvar - var) > np.pi:
                    raise RuntimeError('big jump when computing angle! %f, %f'%(lvar, var))

        last_rpy = adj_rpy

    resp = RequestDMP(u,self.dt,k_gain,d_gain,num_basis)

    self.goal_pose = pose
    self.goal_object_position = goal
    self.dmp_list = resp.dmp_list
    self.tau = resp.tau

  def params(self):
    params = [self.tau,] + list(self.goal_pose.p) + list(self.goal_pose.M.GetQuaternion())
    for dmp in self.dmp_list:
        params += dmp.weights
    print "=========================="
    print "FROM:", params
    print self.tau
    print self.goal_pose
    return params
    
  def _fromParams(self, params):
    '''
    Parse in the cartesian skill from a set of parameters and a config
    '''
    k_gain = self.config['dmp_k']
    d_gain = self.config['dmp_d']
    num_basis = self.config['dmp_basis']
    num_dmps = 6

    self.dmp_list = []

    self.tau = params[0]
    x, y, z, qx, qy, qz, qw = params[1:8]
    self.goal_pose = kdl.Frame(kdl.Rotation.Quaternion(qx, qy, qz, qw))
    self.goal_pose.p[0] = x
    self.goal_pose.p[1] = y
    self.goal_pose.p[2] = z

    print "=========================="
    print "TO:", params
    print self.tau
    print self.goal_pose
    idx = 8
    for i in xrange(num_dmps):
        weights = params[idx:(idx+num_basis+1)]
        self.dmp_list.append(DMPData(
            k_gain=k_gain,
            d_gain=d_gain,
            weights=weights))
        idx += num_basis + 1
