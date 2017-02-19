import numpy as np

" grid "
from features import LoadRobotFeatures

" loading data "
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

" tf/ros utilities "
import tf_conversions.posemath as pm
import PyKDL

'''
LoadDmpData
Go through a list of filenames and load all of them into memory
Also learn a whole set of DMPs
'''
def LoadDmpData(filenames):
    params = []
    data = []
    for filename in filenames:
        print 'Loading demonstration from "%s"'%(filename)
        demo = grid.LoadRobotFeatures(filename)
        
        print "Loaded data, computing features..."
        #fx,x,u,t = demo.get_features([('ee','link'),('ee','node'),('link','node')])
        fx = demo.GetTrainingFeatures()
        x = demo.GetJointPositions()

        print "Fitting DMP for this trajectory..."
        # DMP parameters
        dims = len(x[0])
        dt = 0.1
        K = 100
        D = 2.0 * np.sqrt(K)
        num_bases = 4

        resp = RequestDMP(x,0.1,K,D,5)
        dmp = resp.dmp_list

        dmp_weights = []
        for idmp in dmp:
            dmp_weights += idmp.weights
            num_weights = len(idmp.weights)
        weights += [dmp_weights]

        goals += [x[-1]]
        params += [[i for i in x[-1]] + dmp_weights]

        data.append((demo, fx, dmp))
    return (params, data)

'''
LoadData
Go through a list of filenames and load all of them into memory
Also learn a whole set of DMPs
'''
def LoadData(filenames,objs,manip_objs=[],preset='wam_sim'):
    params = []
    data = []
    goals = []
    for filename in filenames:
        print 'Loading demonstration from "%s"'%(filename)
        demo = LoadRobotFeatures(filename)
        
        print "Loaded data, computing features..."
        #fx,x,u,t = demo.get_features([('ee','link'),('ee','node'),('link','node')])
        demo.UpdateManipObj(manip_objs)
        fx,g = demo.GetTrainingFeatures(objs=objs)
        x = demo.GetJointPositions()

        data.append((demo, fx))
        goals.append(g.squeeze())
    return (data, goals)
