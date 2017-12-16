import os
import numpy as np
# TODO(cpaxton): remove pygame from this
#import pygame as pg

from costar_task_plan.mcts import Node

'''
loop over all MCTS scenarios
- generate the scenarios you need to collect the data
- create 
'''


def mctsLoop(env, policies, seed, save, animate, **kwargs):

    if seed is not None:
        world_id = int(seed)
    else:
        world_id = np.random.randint(10000)
    np.random.seed(world_id)

    env.reset()
    world = env._world
    current_root = Node(world=world)
    done = current_root.terminal

    if policies._rollout is None:
        rollout = "norollout"
    else:
        rollout = "rollout"
    if policies._dfs:
        dfs = "_dfs"
    else:
        dfs = ""
    if policies._sample is not None:
        sample = policies._sample.getName()
    else:
        sample = "none"

    dirname = "world%d_%s_%s%s" % (world_id, sample, rollout, dfs)

    if save or animate:
        window = world._getScreen()
        os.mkdir(dirname)

    while not done:

        # planning loop: determine the set of policies
        for i in xrange(kwargs['iter']):
            # do whatever you want here
            policies.explore(current_root)
        path = policies.extract(current_root)

        # execute loop: follow these policies for however long we are supposed
        # to follow them according to their conditions

        while current_root.state.t < 1.0:
            # compute the next action according to the current policy
            # if a policy is finished, pop it off of the stack
            pass
            done = current_root.terminal
            if animate:
                # show the current window
                pass
            # if save:
            #    # Save pygame image to disk
            #    pg.image.save(window, "%s/iter%d.png"%(dirname,iter))
            if done:
                break

        # update current root
