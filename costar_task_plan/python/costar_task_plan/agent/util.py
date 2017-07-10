
from null import *
from random_agent import *
from task import *

from keras_ddpg import *
from keras_naf import *
from apl_ddpg import *

def GetAgents():
    return ["null", "random", "task", "keras_ddpg", "keras_naf", "apl_ddpg"]

def MakeAgent(env, name, *args, **kwargs):
    try:
        return {
                'null': NullAgent(env, *args, **kwargs),
                'random': RandomAgent(env, *args, **kwargs),
                'task': TaskAgent(env, *args, **kwargs),
        		'keras_ddpg': KerasDDPGAgent(env, *args, **kwargs),
                'apl_ddpg': APLDDPGAgent(env, *args, **kwargs),
                }[name]
    except KeyError, e:
        raise NotImplementedError('Agent "%s" not implemented'%name)

