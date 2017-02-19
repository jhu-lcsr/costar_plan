from dtw import dtw
import numpy as np

" loading data "
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

" message types "
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

" tf/ros utilities "
import tf_conversions.posemath as pm
import PyKDL
import numpy as np

'''
Demonstration
struct to hold robot task demonstration data.

Provides:
    - get_features: computes alignment between different data sources and returns a simplified set of data
'''
class Demonstration:
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

    '''
    get_features returns a task-space view of the demonstration
    and a merged version of the trajectory
    '''
    def get_features(self,frames=[]):

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

    def get_world_pose_msg(self,frame):

        msg = PoseArray()
        msg.header.frame_id = "/world"

        for i in range(len(self.world_t)): 
            pmsg = pm.toMsg(self.tform[frame][i])
            msg.poses.append(pmsg)

        return msg

'''
get_pose_message
Returns a pose array for debugging purposes
'''
def GetPoseMessage(fx, idx, frame_id="/world"):

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

'''
LoadYaml
Really simple function to quickly load from a yaml file
'''
def LoadYaml(filename):
    stream = file(filename,'r')
    demo = yaml.load(stream,Loader=Loader)

    return demo

'''
SaveYaml
Really simple function to quickly load from a yaml file
'''
def SaveYaml(filename,demo):
    stream = file(filename,'w')
    yaml.dump(demo,stream,Dumper=Dumper)

