
import numpy as np

from costar_task_plan.abstract import *

# State of a particular actor. It's the joint state, nice and simple.


class CostarState(AbstractState):

    def __init__(self, world, actor_id, q, dq,
                 finished_last_sequence=False,
                 reference=None,
                 traj=None,
                 seq=0,
                 gripper_closed=False,
                 code=None):

        # Set up list of predicates
        self.predicates = []
        if isinstance(q, list):
            q = np.array(q)
        self.q = q
        self.dq = dq
        self.code = code

        # These are used to tell us which high-level action the robot was
        # performing, and how far along it was.
        self.reference = reference
        self.traj = traj
        self.finished_last_sequence = finished_last_sequence
        self.seq = seq

        # Is the gripper open or closed?
        self.gripper_closed = gripper_closed

        # This should (hopefully) be a reference the world. it hardly matters
        # for something like this, though -- our states hold very little
        # information. The actor ID is important for looking up config information
        # for the actor when updating their state or computing motion plans.
        self.world = world
        self.actor_id = actor_id

    def toArray(self):
        return self.q

# Actions for a particular actor. This is very simple, and just represents a
# joint motion, normalized over some period of time.


class CostarAction(AbstractAction):

    def __init__(self, q, dq, ee=None, reset_seq=False,
                 finish_sequence=False,
                 reference=None,
                 traj=None,
                 gripper_cmd=None,
                 code=None):
        if isinstance(dq, list):
            dq = np.array(dq)
        if isinstance(q, list):
            q = np.array(q)

        self.code = code
        self.q = q
        self.ee = ee
        self.dq = dq
        self.reset_seq = reset_seq
        self.finish_sequence = finish_sequence
        self.reference = reference
        self.traj = traj
        self.gripper_cmd = gripper_cmd

    def toArray(self):
        return self.dq

# This actor represents a robot in the world.
# It's mostly defined by its config -- most of the actual logic that uses this
# is defined in the world's _update_environment() function that gets called after every
# update.


class CostarActor(AbstractActor):

    actor_type = 'robot'

    def __init__(self, config, *args, **kwargs):
        super(CostarActor, self).__init__(*args, **kwargs)
        self.config = config
        self.joints = config['joints']
        self.dof = self.config['dof']
        self.base_link = self.config['base_link']
        if not self.dof == len(self.joints):
            raise RuntimeError('You configured the robot joints wrong')


# Simple policy for these actors
class NullPolicy(AbstractPolicy):

    def evaluate(self, world, state, actor=None):
        return CostarAction(q=state.q, dq=np.zeros(state.q.shape))
