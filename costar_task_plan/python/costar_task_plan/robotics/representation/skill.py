
# ROS stuff
import rospy
from urdf_parser_py.urdf import URDF
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import copy

import numpy as np

# KDL utilities
import PyKDL
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

# machine learning utils (python)
from sklearn.mixture import GMM


# tf stuff
import tf
import tf_conversions.posemath as pm

# input message types 
import sensor_msgs
import trajectory_msgs
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

# output message types
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

from features import RobotFeatures

'''
RobotSkill
Defines a skill with a goal and a set of differential constraints
Goals are represented as the distribution of features that must be true for an action to be considered successful
'''
class RobotSkill:
    
    '''
    set up the robot skill
    skills contain a model of expected features as they change over time
    they also contain a description for our own purposes
    oh, and which objects are involved
    '''
    def __init__(self,data=[],goals=[],action_k=4,goal_k=4,objs=[],manip_objs=[],name="",filename=None,num_gripper_vars=3,normalize=True):
        self.name = name

        self.action_model = GMM(n_components=action_k,covariance_type="full")
        self.goal_model = GMM(n_components=goal_k,covariance_type="full")
        self.trajectory_model = GMM(n_components=1,covariance_type="full")

        # NOTE: gripper stuff is currently not supported. Take a look at this
        # later on if you want to use it.
        self.num_gripper_vars = num_gripper_vars
        self.gripper_model = GMM(n_components=action_k,covariance_type="full")

        self.objs = objs
        self.manip_objs = manip_objs

        if filename == None and len(data) > 0:

            ''' compute means and normalize incoming data '''
	    if normalize:
            	self.action_mean = np.mean(data,axis=0)
            	self.action_std = np.std(data,axis=0)
            	self.goal_mean = np.mean(goals,axis=0)
            	self.goal_std = np.std(goals,axis=0)
	    else:
                self.action_mean = np.ones(data.shape[1])
                self.action_std = np.ones(data.shape[1])
                self.goal_mean = np.ones(data.shape[1])
                self.goal_std = np.ones(data.shape[1])

            goals = (goals - self.goal_mean) / self.goal_std
            data = (data - self.action_mean) / self.action_std

            ''' compute the actual models '''
            # learn action, goal, and trajectory models
            if goal_k > 1:# or True:
                self.goal_model.fit(goals)
            else:
                self.goal_model.means_ = np.array([np.mean(goals,axis=0)])
                self.goal_model.covars_ = np.array([np.cov(goals,rowvar=False)])
                self.goal_model.covars_[0] += 1e-10 * np.eye(self.goal_model.covars_.shape[1])
            if action_k > 1:# or True:
                self.action_model.fit(data)
            else:
                self.action_model.means_ = np.array([np.mean(data,axis=0)])
                self.action_model.covars_ = np.array([np.cov(data,rowvar=False)])
                self.action_model.covars_[0] += 1e-10 * np.eye(self.action_model.covars_.shape[1])

            self.t_factor = 0.1

            if 'gripper' in objs:
                # remove last few indices from
                self.gripper_model = self.action_model
                self.action_model = copy.deepcopy(self.gripper_model)

                # marginalizing out vars in gaussians is easy
                self.action_model.means_ = self.gripper_model.means_[:,:-num_gripper_vars]
                self.action_model.covars_ = self.gripper_model.covars_[:,:-num_gripper_vars,:-num_gripper_vars]
                self.goal_model.covars_ = self.goal_model.covars_[:,:-num_gripper_vars,:-num_gripper_vars]
                self.goal_model.means_ = self.goal_model.means_[:,:-num_gripper_vars]

                self.action_mean_ng = self.action_mean[:-num_gripper_vars]
                self.action_std_ng = self.action_std[:-num_gripper_vars]
                self.goal_mean_ng = self.goal_mean[:-num_gripper_vars]
                self.goal_std_ng = self.goal_std[:-num_gripper_vars]
            else:
                self.action_mean_ng = self.action_mean
                self.action_std_ng = self.action_std
                self.goal_mean_ng = self.goal_mean
                self.goal_std_ng = self.goal_std


        elif not filename == None:
            stream = file(filename,'r')
            data = yaml.load(stream,Loader=Loader)

            self.name = data['name']
            self.action_model = data['action_model']
            self.goal_model = data['goal_model']
            self.gripper_model = data['gripper_model']
            self.trajectory_model = data['trajectory_model']
            self.objs = data['objs']
            self.manip_objs = data['manip_objs']
            self.num_gripper_vars = data['num_gripper_vars']
            self.action_mean = data['action_mean']
            self.action_std = data['action_std']
            self.goal_mean = data['goal_mean']
            self.goal_std = data['goal_std']
            self.action_mean_ng = data['action_mean_ng']
            self.action_std_ng = data['action_std_ng']
            self.goal_mean_ng = data['goal_mean_ng']
            self.goal_std_ng = data['goal_std_ng']
            self.t_factor = 0.1

    def GetGoalModel(self,objs,preset=None):
        if preset is None:
            robot = RobotFeatures()
        else:
            robot = RobotFeatures(preset=preset)

        if 'gripper' in objs:
            objs.remove('gripper')

        for obj in objs:
            robot.AddObject(obj)

        dims = robot.max_index
        K = self.action_model.n_components

        goal = GMM(n_components=K,covariance_type="full")
        goal.weights_ = self.action_model.weights_
        goal.means_ = np.zeros((K,dims))
        goal.covars_ = np.zeros((K,dims,dims))

        idx = robot.GetDiffIndices(objs)
        print objs

        for k in range(K):
            goal.means_[k,:] = self.action_model.means_[k,idx]
            for j in range(dims):
                goal.covars_[k,j,idx] = self.action_model.covars_[k,j,idx]

        return goal

    '''
    save the robot skill to a file
    '''
    def save(self,filename):
        stream = file(filename,'w')

        out = {}
        out['name'] = self.name
        out['action_model'] = self.action_model
        out['goal_model'] = self.goal_model
        out['gripper_model'] = self.gripper_model
        out['trajectory_model'] = self.trajectory_model
        out['objs'] = self.objs
        out['manip_objs'] = self.manip_objs
        out['num_gripper_vars'] = self.num_gripper_vars
        out['action_mean'] = self.action_mean
        out['action_std'] = self.action_std
        out['goal_mean'] = self.goal_mean
        out['goal_std'] = self.goal_std
        out['action_mean_ng'] = self.action_mean_ng
        out['action_std_ng'] = self.action_std_ng
        out['goal_mean_ng'] = self.goal_mean_ng
        out['goal_std_ng'] = self.goal_std_ng

        yaml.dump(out,stream)

    '''
    execution loop update for trajectory skills
    '''
    def update(self, trajs, p_z, t, p_obs=1):
        wts = np.zeros(len(trajs));
        for i in range(len(trajs)):
            p_exp = self.trajectory_model.score(trajs[i][:-1])
            p_exp_f = self.goal_model.score(trajs[i][-1])

            p_exp = np.concatenate((p_exp,p_exp_f))

            wts[i] = weight(p_exp, 1, p_z[i], t[i], self.t_lambda)


'''
This function determines the weight on a given example point
p_expert: different for each point
p_obs: same (actually fixed at 1)
p_z: same for each trajectory
t: different for each point
'''
def weight(p_expert,p_obs,p_z,t,t_lambda=0.1):
    return (p_expert * t_lambda**(1-t)) / (p_obs * p_z)

