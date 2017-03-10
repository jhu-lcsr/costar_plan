
from state import *
from action import *
from dynamics import *

'''
World
A World is a construct that can evolve in time and contains the description
necessary for a particular RL problems.

The important World functions to know about are:
  tick() -- update the current world
  fork(action) -- split off a version of the world assuming the learner takes
                  a particular action at the current time step.
'''
from abstract_world import *

'''
Describe abstract feature generators.
'''
from features import *

'''
Describe abstract qualities of trajectories, and trajectory-related problems.
'''
from trajectory import *


'''
Conditions
These represent predicates that are relevant to our task. In general, we use
these to represent TERMINATION CONDITIONS for our various worlds. Call one with
a world as an argument and it will determine if the condition is no longer true
for that world.

By default, the AbstractWorld holds a list of conditions that fire if they are
violated. Each is associated with a weight.
'''
from condition import *

'''
Option
An option is a sub-policy that will satisfy some intermediate goal.

The Option class also encloses code to specifically train options for various
problems. The important functions here are:
  - makeTrainingWorld(): create a world configured to train this option.
  - makePolicy(): get a policy that will follow this option.
'''
from option import *

'''
Task
This represents the branching tree of possible actions we can take. These are
generally thought to be represented by parameterized controllers that we call
options.
'''
from task import Task


__all__ = ['AbstractWorld', 'AbstractAction', 'Abstractstate',
    'AbstractOption', 'AbstractPolicy', 'AbstractDynamics',
    'Task','AbstractTrainer']

