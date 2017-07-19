
from null import *
from random import *
from task import *

from keras_ddpg import *
from keras_naf import *

def GetAgents():
    return ["none", "null", "albert", "random", "task", "keras_ddpg", "keras_naf"]

def MakeAgent(env, name, *args, **kwargs):
    try:
        return {
                'no': lambda: NoAgent(env, *args, **kwargs),
                'none': lambda: NoAgent(env, *args, **kwargs),
                'null': lambda: NullAgent(env, *args, **kwargs),
		'albert': lambda: AlbertAgent(env, *args, **kwargs),
                'random': lambda: RandomAgent(env, *args, **kwargs),
                'task': lambda: TaskAgent(env, *args, **kwargs),
		'keras_ddpg': lambda: KerasDDPGAgent(env, *args, **kwargs),
                }[name.lower()]()
    except KeyError, e:
        raise NotImplementedError('Agent "%s" not implemented'%name)

