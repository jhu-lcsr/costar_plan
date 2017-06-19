
from null import *
from random import *
from keras_ddpg import *
from keras_naf import *

def GetAgents():
    return ["null", "random", "keras_ddpg", "keras_naf"]

def MakeAgent(name, *args, **kwargs):
    try:
        return {
                'null': NullAgent(*args, **kwargs),
                'random': RandomAgent(*args, **kwargs),
                }[name]
    except KeyError, e:
        raise NotImplementedError('Agent "%s" not implemented'%name)

