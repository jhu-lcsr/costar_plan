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

def MakeSupervisor(x,
        num_actions,
        supervisor_levels=1,
        supervisor_size=128,
        *args,**kwargs):
    '''
    This policy chooses between a number of discrete options representing
    branches in the tree that we could be executing at any given time.

    Parameters:
    -----------
    x: input
    num_actions: number of discrete high level actions to choose from
    supervisor_levels: hidden layers
    supervisor_size: size of hidden layers
    
    Returns:
    --------
    x: encoded set of features
    '''
    for i in xrange(supervisor_levels):
        x = Dense(supervisor_size)(x)
        x = Activation("relu")(x)

    # linear activation for the last layer. This gives us the probability of
    # each of our high-level actions being chosen at a given point in time.
    x = Dense(num_actions)(x)
    
    return x

def MakeCondition(x, x0, num_actions, *args, **kwargs):
    '''
    This policy takes the encoded set of features and predicts whether or not
    that action will continue.

    Parameters:
    -----------
    x: input data
    x0: first frame of the action

    Returns:
    --------
    x: probability that this action is still OK to continue
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
