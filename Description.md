
# Description of Approach

So the goal of this task planner is to combine learning and planning to let us quickly instantiate skills in new environments. Here, skills are defined as the combination of:
  - a parameterizable low level controller
  - a feature function mapping from the world

Our planner functions in two stages:
  - learning: get a mapping between a feature space and parameters.


## Options

Options are control policies (or, alternately, skills). We take advantage of our knowledge of what parametrized policies can/should look like to get these; they are not going to be arbitrary trajectories. For many robots, a policy is going to be either a DMP or a spline.

Now, in general we can perform different options in different ways in different environments.

### Arm Options

The motion of a robotic arm can usually be assumed to start and stop when velocity hits zero. That makes this easy for us, because DMPs are perfect for representing this sort of movement (they can also represent cyclical motions pretty well, but we do not concern ourselves with that).

### Other Options

For something like the needle in Needle Master, we just need a curve or spline to represent individual options. Stopping does not indicate that an option has ended as it does in the manipulation domain above.


