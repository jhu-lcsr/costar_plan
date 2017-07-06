
from null import *
from random import *
from task import *

from keras_ddpg import *
from keras_naf import *

def GetAgents():
    return ["null", "random", "task", "keras_ddpg", "keras_naf"]

def MakeAgent(env, name, *args, **kwargs):
    print "============="
    print name
    try:
        return {
                'null': lambda: NullAgent(env, *args, **kwargs),
                'random': lambda: RandomAgent(env, *args, **kwargs),
                'task': lambda: TaskAgent(env, *args, **kwargs),
		'keras_ddpg': lambda: KerasDDPGAgent(env, *args, **kwargs),
                }[name]()
    except KeyError, e:
        raise NotImplementedError('Agent "%s" not implemented'%name)

