
from costar_task_plan.abstract import AbstractPolicy
from costar_task_plan.robotics.representation import PlanDMP, RequestActiveDMP

# This class takes the planning scene interface from the world and uses
# it to compute (and follow!) a DMP. It will check the current time vs. its
# tick rate before recomputing the DMP.
class DmpPolicy(AbstractPolicy):

  def __init__(self, goal_frame, dmp, kinematics):
    self.dmp = dmp
    self.kinematics = kinematics

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

  def evaluate(self, world, state, actor=None):
    if state.seq > 0 and self.traj is None:
        raise RuntimeError('dmp received bad state')

    # make the trajectory based on the current state
    T = self.kinematics.forward(state.q)

    if state.seq == 0:
        RequestActiveDmp(self.dmp.dmp_list)
        print self.goal_frame
        print world.obs[self.goal_frame]
        print self.dmp.goal_pose
        goal = world.obs[self.goal_frame] * self.dmp.goal_pose
        ee_rpy = T.M.GetRPY()
        rpy = goal.M.GetRPY()
        adj_rpy = [0,0,0]
        for j, (lvar, var) in enumerate(zip(ee_rpy, rpy)):
            if lvar < 0 and var > lvar + np.pi:
                adj_rpy[j] = var - 2*np.pi
            elif lvar > 0 and var < lvar - np.pi:
                adj_rpy[j] = var + 2*np.pi
            else:
                adj_rpy[j] = var
        x = [T.p[0], T.p[1], T.p[2], ee_rpy[0], ee_rpy[1], ee_rpy[2]]
        g = [goal.p[0], goal.p[1], goal.p[2], adj_rpy[0], adj_rpy[1], adj_rpy[2]]
        print "goal = ", g
        x0 = [0.]*6
        g_threshold = [1e-2]*6
        res = PlanDMP(x,x0,0.,g,g_threshold,1.,self.dmp.tau,world.dt,integrate_iter=10)
        print "========"
        print res
        
