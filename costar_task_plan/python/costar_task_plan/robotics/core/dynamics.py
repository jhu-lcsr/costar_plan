
from actor import *

from costar_task_plan.abstract import AbstractDynamics

import rospy
from sensor_msgs.msg import JointState

'''
Listen to the ROS topics, and publish the appropriate commands.
- this one is meant for execution only
- it uses the usual ROS subscriber/publisher setup to accomplish things
'''
class SubscriberDynamics(AbstractDynamics):
  def __init__(self, listener):
    self.listener = listener

  def apply(self, state, action, dt):
    raise NotImplementedError('no subscriber dynamics yet!')

    # publish joint command to whatever topic

    # then update and return
    return CostarState(state.world, q=self.listener.q0)

'''
Apply the motion at each of the joints to get the next point we want to move to
'''
class SimulatedDynamics(AbstractDynamics):
  def __init__(self, config):
    self.q0 = None
    self.dof = config['dof']
    self.old_q0 = [0] * self.dof
    self.dof = self.dof

  def apply(self, state, action, dt):
    q = state.q + (action.dq * dt)
    return CostarState(state.world, q=q, dq=action.dq)


