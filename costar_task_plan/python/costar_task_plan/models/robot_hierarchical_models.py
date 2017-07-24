'''

By Chris Paxton
(c) 2017 The Johns Hopkins University
See license for details

This file defines the architecture for a multilevel robot model, learned via
"divide and conquer" at least at first. It will also let us define the
multilevel models necessary for other things.


[root] --> [A B C], [features]
[A, start_features, features] --> [control, is_ok]
[B, start_features, features] --> [control, is_ok]

Essentially, the high level option is just a normal softmax policy over
low-level options.

Low level options take in [start features, current features].

Note that we always assume this is a purely supervised problem -- we know
exactly when actions should start, when they should stop, and training data
represents all of this.
'''

from robot_multi_models import *
   
def MakeSupervisor(labels,*args,**kwargs):
    '''
    This policy chooses between a number of discrete options representing
    branches in the tree that we could be executing at any given time.

    Parameters:
    -----------
    labels: set of integer labels indicating which OPTION each control was
    assigned to.
    
    Returns:
    --------
    ins: inputs
    x: encoded set of features
    '''
    x = GetEncoder(*args, **kwargs)
    
    pass

def MakeCondition(x, *args, **kwargs):
    '''
    This policy takes the encoded set of features and predicts whether or not
    that action will continue.
    '''
    pass

def MakeControlPolicy(x, policy_dense_size,
        num_dense_layers=1,
        arm_size=6,
        gripper_size=1,
        *args, **kwargs):
    '''
    This policy takes in the encoded feature set and predicts an immediate
    motion goal to be sent to the controllers.
    '''
    for i in xrange(num_dense_layers):
        x = Dense(policy_dense_size, x)
        x = LeakyReLU(alpha=0.2)(x)
    arm = Dense(arm_size)
    gripper = Dense(gripper_size)
    return [arm, gripper]

def MakePredictionPolicy(x, *args, **kwargs):
    '''
    This policy takes in the encoded set of features and predicts the next set
    of features.
    '''
    pass
