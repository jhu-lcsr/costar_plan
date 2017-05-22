
from random import *
from keras_ddpg import *
from keras_naf import *

def GetAgents():
    return ["random", "keras_ddpg", "keras_naf"]

def MakeAgent(name, *args, **kwargs):
    try:
        return {
                'random': RandomAgent(*args, **kwargs),
                }[name]
    except KeyError, e:
        print 'Agent "%s" not implemented'%name
