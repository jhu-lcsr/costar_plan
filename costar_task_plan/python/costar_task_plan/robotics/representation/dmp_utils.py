" grid "
from features import LoadRobotFeatures

" ros utils "
import rospy
import copy

" numpy "
import numpy as np

" dmp message types "
from dmp.msg import *
from dmp.srv import *

"""
DMP UTILITIES
===============================================================
These are based on the sample code from http://wiki.ros.org/dmp
(sample code by Scott Niekum)
"""

'''
Put together a DMP request
'''
def RequestDMP(u,dt,k_gain,d_gain,num_basis_functions):

    ndims = len(u[0])
    k_gains = [k_gain]*ndims
    d_gains = [d_gain]*ndims
    
    ex = DMPTraj()
    
    for i in range(len(u)):
        pt = DMPPoint()
        pt.positions = u[i] # always sends positions regardless of actual content
        ex.points.append(pt)
        ex.times.append(dt * i) # make sure times are reasonable

    print "Waiting for DMP service..."
    rospy.wait_for_service('learn_dmp_from_demo')

    print "Sending DMP learning request..."
    try:
        lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
        resp = lfd(ex, k_gains, d_gains, num_basis_functions)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    print "LfD done"    
            
    return resp;

def PlanDMP(x_0, x_dot_0, t_0, goal, goal_thresh, 
                    seg_length, tau, dt, integrate_iter):

    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, 
                   seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    return resp;

def RequestActiveDMP(dmps):
    try:
        sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
        sad(dmps)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

"""
Other Utilities
================================================================
These are by Chris Paxton, used for loading data and searching
for trajectories.
"""

'''
LoadDataDMP
Go through a list of filenames and load all of them into memory
Also learn a whole set of DMPs
'''
def LoadDataDMP(filenames,objs,manip_objs=[],preset='wam_sim'):
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

        print "Fitting DMP for this trajectory..."
        # DMP parameters
        dims = len(x[0])
        dt = 0.1
        K = 100
        D = 2.0 * np.sqrt(K)
        num_bases = 5

        resp = RequestDMP(x,0.1,K,D,5)
        dmp = resp.dmp_list

        dmp_weights = []
        for idmp in dmp:
            dmp_weights += idmp.weights
            num_weights = len(idmp.weights)

        params += [[i for i in x[-1]] + dmp_weights]

        data.append((demo, fx, resp))
        goals.append(g.squeeze())
    return (data, params, num_weights, goals)

'''
ParamFromDMP
Get a vector of floating-point numbers from the DMP representing the
trajectory parameterization.
'''
def ParamFromDMP(goal,dmp):
    dmp_weights = []
    for idmp in dmp:
        dmp_weights += [i for i in idmp.weights]
    params = [i for i in goal] + dmp_weights

    return params

'''
ParamToDMP
Take a vector of parameters and turn it into a DMP.
'''
def ParamToDMP(param,dmp,dims=7,num_weights=6):
    dmp2 = copy.deepcopy(dmp)
    goal = param[:dims]
    for j in range(dims):
        idx0 = dims + (j * num_weights)
        idx1 = dims + ((j+1) * num_weights)
        dmp2[j].weights = param[idx0:idx1]

    return (goal, dmp2)


'''
SearchDMP
'''
def SearchDMP(Z,robot,world,
        x0,xdot0,t0,threshold,seg_length,tau,dt,int_iter,dmp,
        ll_percentile=90,
        num_weights=6,
        num_samples=100):

    #print "Search iteration:"
    dmps = Z.sample(100)

    assert(len(world.keys())==1)

    #search = MarkerArray()
    gen_trajs = []
    gen_params = []
    lls = []
    count = 1
    for i in range(100):
        dmp2 = copy.deepcopy(dmp)
        (goal,dmp2) = ParamToDMP(dmps[i],dmp,num_weights=num_weights)

        RequestActiveDMP(dmp2)
        plan = PlanDMP(x0,xdot0,t0,goal,threshold,seg_length,tau,dt,int_iter)

        #ll = robot.GetTrajectoryLikelihood(plan.plan.points,world)
        #print ll

        traj = [pt.positions[:7] for pt in plan.plan.points]
        ll = robot.GetTrajectoryLikelihood(traj,world,objs=['link'])
        #ll = robot.GetTrajectoryLikelihood(traj,world,(10,17))
        #ll = robot.GetTrajectoryLikelihood(traj,world,(3,3+(7*len(world.keys()))))

        lls.append(ll)

        gen_trajs.append(traj)
        gen_params.append(ParamFromDMP(goal,dmp2))

    search_lls = []
    search_trajs = []
    search_params = []

    #print "... done with DMPs."

    search_params = []
    ll_threshold = np.percentile(lls,90)
    for (ll,param,traj) in zip(lls,gen_params,gen_trajs):
        if ll > ll_threshold:
            search_params.append(param)
            search_trajs.append(traj)
            search_lls.append(ll)


    print "... Done. Average goal probability: %f"%(np.mean(lls))
    print "    Found %d with p>%f."%(len(search_params),ll_threshold)

    return lls,search_lls,search_trajs,search_params,gen_trajs

