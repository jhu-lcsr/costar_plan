from costar_task_plan.mcts.search import RandomSearch

def OptimizePolicy(world, task, policies, num_iter=100, num_samples=25, *args, **kwargs):
    '''
    Run a cross-entropy like sampling loop to optimize some parameterized
    policy.

    - samples a single stochastic trace through the world
    '''
    for i in xrange(num_iter):
        # Number of iterations for policy search
        for j in xrange(num_samples):
            # Forward pass: draw samples from the associated policy until
            # completion.
            reward, traj = ForwardPass(world, task, policies)


def ForwardPass(world, task, policies):
    world = world.duplicate()
    task.compile(world)
    root = Node(world=world,root=True)
    search = RandomSearch()
    t, path = search(root)
    print "time =", t
    print path
