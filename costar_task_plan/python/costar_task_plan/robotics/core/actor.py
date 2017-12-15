
import numpy as np

from costar_task_plan.abstract import *

from .js_listener import JointStateListener

class CostarState(AbstractState):

    '''
    State of a particular actor. It's the joint state, nice and simple, but
    it also includes some other information.
    '''

    def __init__(self, actor_id, q, dq,
                 finished_last_sequence=False,
                 reference=None,
                 traj=None,
                 seq=0,
                 gripper_closed=False,
                 t=0.,
                 code=None):

        # Set up list of predicates
        self.predicates = []
        if isinstance(q, list):
            q = np.array(q)
        self.q = q
        self.dq = dq
        self.code = code
        self.t = t

        # These are used to tell us which high-level action the robot was
        # performing, and how far along it was.
        self.reference = reference
        self.traj = traj
        self.finished_last_sequence = finished_last_sequence
        self.seq = seq

        # Is the gripper open or closed?
        self.gripper_closed = gripper_closed

        # Identity of the particular actor
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
                 code=None,
                 error=False,):
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
        self.error = error

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
        self.js_listener = JointStateListener(self.config)
        self.name = config['name']
        self.joints = config['joints']
        self.dof = self.config['dof']
        self.base_link = self.config['base_link']
        if not self.dof == len(self.joints):
            raise RuntimeError('You configured the robot joints wrong')

    def getState(self):
        '''
        Get and update the actor's state
        '''
        q = self.js_listener.q0
        dq = self.js_listener.dq
        return CostarState(self.id, q, dq)


# Simple policy for these actors
class NullPolicy(AbstractPolicy):

    def evaluate(self, world, state, actor=None):
        return CostarAction(q=state.q, dq=np.zeros(state.q.shape))
