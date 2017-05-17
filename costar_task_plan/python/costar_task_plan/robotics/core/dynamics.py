
from actor import *

from costar_task_plan.abstract import AbstractDynamics

import rospy
from sensor_msgs.msg import JointState


class SubscriberDynamics(AbstractDynamics):

    '''
    Listen to the ROS topics, and publish the appropriate commands.
    - this one is meant for execution only
    - it uses the usual ROS subscriber/publisher setup to accomplish things
    '''

    def __init__(self, listener):
        self.listener = listener

    def apply(self, state, action, dt):
        '''
        This should publish joint information to whatever topic, then update
        and return.
        '''
        raise NotImplementedError('no subscriber dynamics yet!')

        return CostarState(state.world, state.actor_id, q=self.listener.q0)


class SimulatedDynamics(AbstractDynamics):

    '''
    Apply the motion at each of the joints to get the next point we want to
    move to. This will also update the gripper state if it's one of the basic
    gripper operations, i.e. gripper_cmd in {open, close}.
    '''

    def __init__(self):
        pass

    def apply(self, state, action, dt):
        '''
        Simulated dynamics assume we can get from our current state to the
        next set point from a DMP trajectory, or whatever.
        '''
        if action.reset_seq \
            or action.reference is not state.reference \
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
                raise RuntimeError(
                    'Unrecognized gripper command: "%s"' % (str(action.gripper_cmd)))
        else:
            gripper_closed = state.gripper_closed

        if state.q.shape == action.q.shape:
            # If we got a position just assume that the action gets there
            # successfully.
            q = action.q
        else:
            # Use the provided velocities to compute a position based on the
            # time stamp.
            q = state.q + (action.dq * dt)

        if action.traj is None:
            traj = state.traj
        else:
            traj = action.traj

        # Costar states also include some state information for the sake of our
        # dynamic movement primitives.
        return CostarState(state.world,
                           state.actor_id,
                           q=q,
                           dq=action.dq,
                           seq=seq,
                           gripper_closed=gripper_closed,
                           finished_last_sequence=action.finish_sequence,
                           traj=traj,
                           reference=action.reference)
