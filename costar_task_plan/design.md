# Design Philosophy

We have a modular architecture that describes a problem (a "world"), with its associated constraints ("conditions"), reward function, features, etc. Worlds are associated with Actors that behave according to different Policies. The key problem TTS tries to solve is to explore the space of possible Policies that may lead to successful results.

Most of the high-level types are implemented in `task_tree_search.abstract`. In general, we try to enforce typing as much as possible in python; this makes it easier to abstract out different functionality, among other things. For example, most Reward functions inherit from `AbstractReward`, and most conditions from `AbstractCondition`; in truth, these things are just classes with a provided `__call__()` operator.

## Environment Design

All environments extend the `AbstractWorld` class. An `AbstractWorld` contains actors, states, and references to all associated conditions and other things like that.

The most important functions are the `tick()` and `fork()` functions associated with the World. The `tick()` function updates a particular world trace; the `fork()` function will create a new world trace.

Pretty much everything here is modular. An example of why this is very important is in the `LateralDynamics` class: we can handle different speeds more or less simply to make our learning problem more reasonable by thesholding velocities around zero to make sure the car actually stops.

### Overview of Environment Classes

  - World: holds the world state
  - Condition: must be true for execution to continue.
  - Reward: function that produces a reward signal given a world.
  - Features: function that extracts a feature vector given a world state.
  - Actor: the entity that is actually performing an action. Has a unique id.
  - State: abstract representation of the current state of a particular actor; not the world state!
  - Action: abstract representation of an action.
  - Dynamics: takes a (world, state, action) and maps to a new state.
  - Policy: takes a (world, state, (optional) actor), and returns a new action.
  - Option: a "sub-problem" that is a key part of a larger problem. Technically, we use these to _learn_ options (in the formal sense): our Policies are the sMDP options.

## Tree Search Design

Goal: explore the set of possible combinations of policies to find optimal paths that satisfy the constraints given by the world.

### Overview of Tree Search Classes

  - Node: contains a World. Nodes have children. New Nodes are created with a combination of `world.fork()` and `world.tick()`.
  - MCTS Policies: contains your basic MCTS operations, like `select()` and `rollout()`.
  - MCTS Action: by default, wraps a policy that governs how the actor should behave over time.
  - MCTS Sampler: produces the set of actions available from a particular world state.

## Learning Design

Goal: find policies that satisfy the constraints expressed by the list of conditions belonging to the current world.

In general, Trainers correspond to Agents in Keras-RL. They implement various algorithms, and are a little broader.

Adversaries are new -- these govern how we choose different worlds. Their goal is to learn policies that can handle outliers.

### Overview of Learning Classes

  - Model -- refers to a Keras/TensorFlow model.
  - Trainer -- Uses various instantiations of an Option to train a particular Model using a certain method.
  - Adversary -- Determines how to instantiate an Option and create a World.
  - Oracle -- Stores, manages, and generates new training data. The PolicyOracle class is the clearest and most general example of this: it just takes a policy and sees what it will do.

### Overview of OpenAI Gym Environments

  - RoadWorldOptionEnv: Train a policy for a particular option.
  - RoadWorldDiscreteSamplerEnv: Use trained options to learn policy for MCTS tree expansion.

