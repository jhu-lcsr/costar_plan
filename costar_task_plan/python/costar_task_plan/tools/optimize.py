
from costar_task_plan.mcts import Node
from costar_task_plan.mcts.search import RandomSearch

def OptimizePolicy(world, policies, num_iter=100, num_samples=25, *args, **kwargs):
    '''
    Run a cross-entropy like sampling loop to optimize some parameterized
    policy.

    - samples a single stochastic trace through the world
    '''
    search = RandomSearch(policies)
    for i in xrange(num_iter):
        # Number of iterations for policy search
        for j in xrange(num_samples):
            # Forward pass: draw samples from the associated policy until
            # completion.
            reward, traj = ForwardPass(world, search)
            print reward


def ForwardPass(world, search):
    '''
    Complete a single optimization forward pass.
    '''
    root = Node(world=world, root=True)
    t, path = search(root)

    return path[-1].reward, path
