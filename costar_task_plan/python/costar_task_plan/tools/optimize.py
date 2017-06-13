
from costar_task_plan.mcts import Node
from costar_task_plan.mcts.search import RandomSearch

def OptimizePolicy(world, task, policies, num_iter=100, num_samples=25, *args, **kwargs):
    '''
    Run a cross-entropy like sampling loop to optimize some parameterized
    policy.

    - samples a single stochastic trace through the world
    '''
    print world.features
    print world.reward
    for i in xrange(num_iter):
        # Number of iterations for policy search
        for j in xrange(num_samples):
            # Forward pass: draw samples from the associated policy until
            # completion.
            reward, traj = ForwardPass(world, task, policies)


def ForwardPass(world, task, policies):
    '''
    Complete a single optimization forward pass.
    '''
    root = task.makeRoot(world)
    search = RandomSearch(policies)
    t, path = search(root)

    return path[-1].reward, path
