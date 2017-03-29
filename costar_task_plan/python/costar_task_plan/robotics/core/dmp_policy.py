
from costar_task_plan.abstract import AbstractPolicy

# This class takes the planning scene interface from the world and uses
# it to compute (and follow!) a DMP. It will check the current time vs. its
# tick rate before recomputing the DMP.
class DmpPolicy(AbstractPolicy):

  def __init__(self, features=None, dmp=None):
    raise NotImplementedError('DMP policy not set up!')

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
    raise NotImplementedError('DMP policy not set up!')

  def evaluate(self, world, state, actor=None):
    raise NotImplementedError('DMP policy not set up!')

