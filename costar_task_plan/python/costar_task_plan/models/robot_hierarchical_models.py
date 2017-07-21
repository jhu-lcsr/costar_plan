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
'''

from robot_multi_models import *

def MakeInternalPolicy(labels):
    '''
    This policy chooses between a number of discrete options.

    Parameters:
    -----------
    labels: set of integer labels indicating which OPTION each control was
    assigned to.
    
    '''
    pass
