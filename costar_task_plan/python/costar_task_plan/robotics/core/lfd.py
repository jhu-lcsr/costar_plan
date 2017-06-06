
from dmp_policy import CartesianDmpPolicy
from dmp_option import DmpOption
from dynamics import SimulatedDynamics
from geometry_msgs.msg import PoseArray
from os.path import join
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf_conversions import posemath as pm
from urdf_parser_py.urdf import URDF

# Include representations for either (a) the generic skill or (b) an instance
# of a skill represented as a goal-directed set of motion primitives.
from costar_task_plan.robotics.representation import RobotFeatures
from costar_task_plan.robotics.representation import CartesianSkillInstance
from costar_task_plan.robotics.representation import GMM
from costar_task_plan.robotics.representation import RequestActiveDMP, PlanDMP

# Important libraries
import numpy as np
import os
import rospy
import yaml

try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper


class LfD(object):

    '''
    Computes features and representations which given trajectories, objects, etc.
    This must take a world object to use for computing objects and positions.
    Note that this LfD object currently assumes supervised (labeled) data, which
    indicates exactly which objects are important for each action and which are
    not.
    '''

    def __init__(self, world):
        self.world = world

        config = world.actors[0].config
        base_link = config['base_link']
        end_link = config['end_link']

        if 'robot_description_param' in config:
            self.robot = URDF.from_parameter_server(
                config['robot_description_param'])
        else:
            self.robot = URDF.from_parameter_server()

        self.config = config
        self.base_link = base_link

        # set up kinematics stuff
        self.tree = kdl_tree_from_urdf_model(self.robot)
        self.chain = self.tree.getChain(base_link, end_link)
        self.kdl_kin = KDLKinematics(self.robot, base_link, end_link)

        self.base_link = base_link
        self.end_link = end_link

        self.skill_instances = {}
        self.skill_features = {}
        self.skill_models = {}

        self.pubs = {}

    def train(self):
        '''
        Generate DMPs and GMMs associated with different labeled actions.
        '''

        print "Training:"
        for name, trajs in self.world.trajectories.items():

            self.pubs[name] = rospy.Publisher(
                join('costar', 'lfd', name), PoseArray, queue_size=1000)

            data = self.world.trajectory_data[name]
            features = RobotFeatures(self.config, self.kdl_kin)
            objs = self.world.objs[name]

            self.skill_instances[name] = []

            for traj, world in zip(trajs, data):

                ts = [t for t, _, _, _ in traj]
                dt = np.mean(np.diff(ts))

                ee = [pm.fromMsg(pose) for _, pose, _, _ in traj]
                gripper = [gopen for _, _, gopen, _ in traj]

                if not len(ee) == len(gripper) and len(gripper) == len(world):
                    raise RuntimeError('counting error')

                # compute features?
                f, g = features.GetFeaturesForTrajectory(ee, world[0], objs)
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

            # only fit models if we have an example of that skill
            if name in self.skill_features:
                self.skill_models[name] = GMM(
                    self.config['gmm_k'], self.skill_features[name])
                print "> Skill", name, "extracted with dataset of shape", self.skill_features[name].shape, "k = ", self.config['gmm_k']
            else:
                print " ... skipping skill", name, "(no data)"

        return self.skill_models

    def debug(self, world, args):
        '''
        Publish a bunch of ROS messages showing trajectories. This is a helpful
        tool for debugging problems with training data, DMP learning, and DMP
        segmentation.
        '''

        for name, instances in self.skill_instances.items():

            model = self.skill_models[name]
            goal_type = instances[0].objs[0]

            option = DmpOption(
                CartesianDmpPolicy, self.kdl_kin, args[goal_type], name, model, instances)
            policy = option.makePolicy(world)
            dynamics = SimulatedDynamics()
            condition = option.getGatingCondition()

            state = world.actors[0].state

            goal = args[goal_type]
            RequestActiveDMP(instances[0].dmp_list)
            goal = world.observation[goal]

            q = state.q
            q = [-0.73408591, -1.30249417,  1.53612047,
                 -2.0823833,   2.29921898,  1.42712378]
            T = pm.fromMatrix(self.kdl_kin.forward(q))
            ee_rpy = T.M.GetRPY()
            relative_goal = goal * instances[0].goal_pose
            rpy = relative_goal.M.GetRPY()
            adj_rpy = [0, 0, 0]

            # Fix rotations
            for j, (lvar, var) in enumerate(zip(ee_rpy, rpy)):
                if lvar < 0 and var > lvar + np.pi:
                    adj_rpy[j] = var - 2 * np.pi
                elif lvar > 0 and var < lvar - np.pi:
                    adj_rpy[j] = var + 2 * np.pi
                else:
                    adj_rpy[j] = var

            # Create start and goal poses
            x = [T.p[0], T.p[1], T.p[2], ee_rpy[0], ee_rpy[1], ee_rpy[2]]
            g = [relative_goal.p[0], relative_goal.p[1], relative_goal.p[2],
                 adj_rpy[0], adj_rpy[1], adj_rpy[2]]
            x0 = [0.] * 6
            g_threshold = [1e-1] * 6
            integrate_iter = 10

            # Get DMP result
            res = PlanDMP(x, x0, 0., g, g_threshold, 2 * instances[
                          0].tau, 1.0, world.dt, integrate_iter)

            # Convert to poses
            poses = []
            for i, pt in enumerate(res.plan.points):
                T = pm.Frame(
                    pm.Rotation.RPY(pt.positions[3], pt.positions[4], pt.positions[5]))
                T.p[0] = pt.positions[0]
                T.p[1] = pt.positions[1]
                T.p[2] = pt.positions[2]
                poses.append(pm.toMsg(T))
                new_q = self.kdl_kin.inverse(pm.toMatrix(T), q)
                q = new_q

            msg = PoseArray(poses=poses)
            msg.header.frame_id = self.base_link
            self.pubs[name].publish(msg)

    def save(self, project_name):
        '''
        Save models after we fit them. Models for skill instances and for
        expected feature observations are recorded separately.
        '''
        skills_dir = os.path.join(project_name, 'skill_models')
        models_dir = os.path.join(project_name, 'feature_models')
        if not os.path.exists(project_name):
            os.mkdir(project_name)
        if not os.path.exists(skills_dir):
            os.mkdir(skills_dir)
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        skill_counts = {}

        for name in self.skill_instances.keys():
            skill_counts[name] = len(self.skill_instances[name])
            for i, skill in enumerate(self.skill_instances[name]):
                filename = os.path.join(skills_dir, '%s%02d.yml'%(name,i))
                yaml_save(skill, filename)
            model_filename = os.path.join(models_dir, '%s_gmm.yml'%name)
            yaml_save(self.skill_models[name], model_filename)

        skill_filename = os.path.join(project_name, "skills.yml")
        yaml_save(skill_counts, skill_filename)

    def load(self, project_name):
        '''
        Load saved models into this object.
        '''
        skills_dir = os.path.join(project_name, 'skill_models')
        models_dir = os.path.join(project_name, 'feature_models')
        skill_filename = os.path.join(project_name, "skills.yml")
        skills = yaml_load(skill_filename)
        for name, count in skills.items():

            # For debugging only
            if name not in self.pubs:
                self.pubs[name] = rospy.Publisher(
                    join('costar', 'lfd', name), PoseArray, queue_size=1000)

            self.skill_instances[name] = []
            for i in xrange(count):
                filename = os.path.join(skills_dir, '%s%02d.yml'%(name,i))
                dmp = yaml_load(filename)
                self.skill_instances[name].append(dmp)

            model_filename = os.path.join(models_dir, '%s_gmm.yml'%name)
            self.skill_models[name] = yaml_load(model_filename)

def yaml_save(obj, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(obj, outfile, default_flow_style=False)

def yaml_load(filename):
    with open(filename, 'r') as infile:
        return yaml.load(infile)
