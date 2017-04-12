
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

# Apply the motion at each of the joints to get the next point we want to move
# to. This will also update the gripper state if it's one of the basic gripper
# operations, i.e. gripper_cmd in {open, close}.
class SimulatedDynamics(AbstractDynamics):

  def __init__(self):
    pass

  def apply(self, state, action, dt):
    if action.reset_seq or action.reference is not state.reference \
            or action.reference is None:
      seq = 0
    else:
      seq = state.seq + 1

    if action.gripper_cmd is not None:
      if action.gripper_cmd == "close":
        gripper_closed = True
      elif action.gripper_cmd == "open":
        gripper_closed = False
      else:
        raise RuntimeError('Unrecognized gripper command: "%s"'%(str(action.gripper_cmd)))

    q = state.q + (action.dq * dt)
    return CostarState(state.world,
            q=q,
            dq=action.dq,
            seq=seq,
            reference=action.reference)


