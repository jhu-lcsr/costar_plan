
from actor import CostarAction

import numpy as np
import rospy
import tf_conversions.posemath as pm

# dmp message types
from dmp.srv import *

from costar_task_plan.abstract import AbstractPolicy
from costar_task_plan.robotics.representation import PlanDMP, RequestActiveDMP

# This class takes the planning scene interface from the world and uses
# it to compute (and follow!) a DMP. It will check the current time vs. its
# tick rate before recomputing the DMP.
class DmpPolicy(AbstractPolicy):

  def __init__(self, goal, dmp, kinematics):
    self.dmp = dmp
    self.kinematics = kinematics
    self.goal = goal
    self.activate = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
    self.plan = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)

  def evaluate(self, world, state, actor=None):
    raise NotImplementedError('DMP policy not set up!')
        
# DMP instance used if we are interested in a joint-space motion.
class JointDmpPolicy(DmpPolicy):

  # This needs to instantiate slightly different things from the other one. In
  # general it's not going to need to compute joint motions itself -- it can
  # just compute a joint difference.
  def __init__(self, *args, **kwargs):
    super(JointDmpPolicy, self).__init__(*args, **kwargs)
    raise NotImplementedError('DMP policy not set up!')

  def evaluate(self, world, state, actor=None):
    raise NotImplementedError('DMP policy not set up!')

# DMP instance used if we are interested in describing cartesian movements. In
# this case, we are going to compute inverse kinematics using PyKDL. We can
# then use these to compute the actual commands that get sent to the robot's
# joints -- since all robot actions take the form of joint positions,
# velocities, and efforts.
class CartesianDmpPolicy(DmpPolicy):

  # This needs to instantiate slightly different things from the other one. In
  # general it's not going to need to compute joint motions itself -- it can
  # just compute a joint difference.
  def __init__(self, *args, **kwargs):
    super(CartesianDmpPolicy, self).__init__(*args, **kwargs)
    self.traj = None
    #self.q = None

  def evaluate(self, world, state, actor=None):
    # =========================================================================
    # make the trajectory based on the current state
    reset_seq = state.reference is not self
    #print "reset?", (state.reference is not self), state.reference, self
    g = []
    if state.seq == 0 or reset_seq:
        T = pm.fromMatrix(self.kinematics.forward(state.q))
        self.activate(self.dmp.dmp_list)
        goal = world.observation[self.goal]
        ee_rpy = T.M.GetRPY()
        relative_goal = goal * self.dmp.goal_pose
        rpy = relative_goal.M.GetRPY()
        adj_rpy = [0,0,0]
        for j, (lvar, var) in enumerate(zip(ee_rpy, rpy)):
            if lvar < 0 and var > lvar + np.pi:
                adj_rpy[j] = var - 2*np.pi
            elif lvar > 0 and var < lvar - np.pi:
                adj_rpy[j] = var + 2*np.pi
            else:
                adj_rpy[j] = var
        x = [T.p[0], T.p[1], T.p[2], ee_rpy[0], ee_rpy[1], ee_rpy[2]]
        g = [relative_goal.p[0], relative_goal.p[1], relative_goal.p[2],
                adj_rpy[0], adj_rpy[1], adj_rpy[2]]
        x0 = [0.]*6
        g_threshold = [1e-1]*6
        integrate_iter=10
        res = self.plan(x,x0,0.,g,g_threshold,self.dmp.tau,1.0,world.dt,integrate_iter)
        q = state.q
        self.traj = res.plan
        #self.q = state.q

    # =========================================================================
    # Compute the joint velocity to take us to the next position
    if state.seq < len(self.traj.points):
      pt = self.traj.points[state.seq]

      T = pm.Frame(pm.Rotation.RPY(pt.positions[3],pt.positions[4],pt.positions[5]))
      T.p[0] = pt.positions[0]
      T.p[1] = pt.positions[1]
      T.p[2] = pt.positions[2]
      q = self.kinematics.inverse(pm.toMatrix(T), state.q)
      #q = self.kinematics.inverse(pm.toMatrix(T), self.q)
      #print "======="
      #print q
      if q is not None:
        #self.q = q
        dq = (q - state.q) / world.dt
        return CostarAction(dq=dq, reset_seq=reset_seq, reference=self)
      else:
        print "!!!!!!!!!!!!!"
        return None
    else:
        action = world.zeroAction()
        action.reference = self
        return action
        

        
