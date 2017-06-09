
from dtw import dtw
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

import collections
import numpy as np
import PyKDL
import tf_conversions.posemath as pm
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

Distribution = collections.namedtuple('Distribution','mu sigma')

class Demonstration:
    '''
    Demonstration
    struct to hold robot task demonstration data.

    Provides:
        - getFeatures: computes alignment between different data sources and returns a simplified set of data
    '''
    def __init__(self,filename=None):
        self.joint_p = []
        self.joint_v = []
        self.joint_t = []
        self.gripper_cmd = []
        self.gripper_t = []
        self.tform = {}
        self.world_t = []

        if not filename==None:
            stream = file(filename,'r')
            demo = yaml.load(stream,Loader=Loader)
            self.joint_p = demo.joint_p
            self.joint_v = demo.joint_v
            self.joint_t = demo.joint_t
            self.gripper_cmd = demo.gripper_cmd
            self.tform = demo.tform
            self.world_t = demo.world_t

    def getFeatures(self,frames=[]):
        '''
        getFeatures returns a task-space view of the demonstration
        and a merged version of the trajectory
        '''

        # make times into floats
        jt = [t.to_sec() for t in self.joint_t]
        gt = [t.to_sec() for t in self.gripper_t]
        wt = [t.to_sec() for t in self.world_t]

        jidx = []
        widx = []

        # select subsamples of times/other variables to create training data
        jd,jc,jpath = dtw(gt,jt)
        wd,wc,wpath = dtw(gt,wt)
        for i in range(len(gt)):
            jidx.append(np.argmin(jc[i,:]))
            widx.append(np.argmin(wc[i,:]))
        
        jp = [self.joint_p[i] for i in jidx]
        jv = [self.joint_v[i] for i in jidx]
        jts = [self.joint_t[i] for i in jidx]
        wts = [self.world_t[i] for i in widx]
        
        # go through all these variables and choose the right ones
        fx = []
        x = []
        u = []
        for i in range(len(gt)):
            features = [(i+1)/float(len(gt))] + self.gripper_cmd[i][0:3] # only include the first 3 fields
            for frame1,frame2 in frames:
                f1 = self.tform[frame1][widx[i]]
                #f2 = self.tform[frame2][widx[i]]
                f2 = self.tform[frame2][0]
                transform = f2.Inverse() * f1

                features += transform.p
                features += transform.M.GetRPY()
                features += [transform.p.Norm()]
            fx.append(features)
            x.append(jp[i] + self.gripper_cmd[i])
            u.append(jv[i] + self.gripper_cmd[i])

        return fx, x, u, gt

    def getWorldPoseMsg(self,frame):
        '''
        Return pose array message in the world frame.
        '''

        msg = PoseArray()
        msg.header.frame_id = "/world"

        for i in range(len(self.world_t)): 
            pmsg = pm.toMsg(self.tform[frame][i])
            msg.poses.append(pmsg)

        return msg

def GetPoseMessage(fx, idx, frame_id="/world"):
    '''
    get_pose_message
    Returns a pose array for debugging purposes
    '''

    msg = PoseArray()
    msg.header.frame_id = frame_id

    for i in range(len(fx)):

        pose = PyKDL.Vector(fx[i][idx],fx[i][idx+1],fx[i][idx+2]);
        #rot = PyKDL.Rotation.RPY(fx[i][idx+3],fx[i][idx+4],fx[i][idx+5])
        q = np.array([fx[i][idx+3],fx[i][idx+4],fx[i][idx+5],fx[i][idx+6]])
        q = q / np.linalg.norm(q)
        rot = PyKDL.Rotation.Quaternion(q[0],q[1],q[2],q[3])
        frame = PyKDL.Frame(PyKDL.Rotation.RotY(np.pi/2)) * PyKDL.Frame(rot,pose) * PyKDL.Frame(PyKDL.Rotation.RotZ(-1*np.pi/2)) * PyKDL.Frame(PyKDL.Rotation.RotY(-1*np.pi/2))
        #frame = PyKDL.Frame(rot,pose) * PyKDL.Frame(PyKDL.Rotation.RotZ(-1*np.pi/2)) #* PyKDL.Frame(PyKDL.Rotation.RotX(-1*np.pi/2))
        pmsg = pm.toMsg(frame)
        msg.poses.append(pmsg)

    return msg

def LoadYaml(filename):
    '''
    LoadYaml
    Really simple function to quickly load from a yaml file
    '''

    stream = file(filename,'r')
    demo = yaml.load(stream,Loader=Loader)

    return demo

def SaveYaml(filename,demo):
    '''
    SaveYaml
    Really simple function to quickly load from a yaml file
    '''

    stream = file(filename,'w')
    yaml.dump(demo,stream,Dumper=Dumper)

