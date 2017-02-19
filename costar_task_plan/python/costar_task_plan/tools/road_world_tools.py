
from road_world_util import *
from task_tree_search.road_world.config import *
from road_world_agents import *

from task_tree_search.gym import RoadWorldOptionEnv
from task_tree_search.models import FastNetwork

from task_tree_search.road_world.core import LateralManualPolicy
from task_tree_search.road_world.ltl import LTLCondition

def evaluate_road_world_option(
    option,
    agent_name,
    test_episodes=1000,
    test_length=1000,
    ltl = True,
    visualize = False,
    lateral = False,
    hz=10,
    verbose = False,
    seed = None,
    use_fast = False,
    baseline = False,
    override = False,
    *args, **kwargs):


  if override:
    env_option = "Planning"
  else:
    env_option = option

  env = RoadWorldOptionEnv(
          option=env_option,
          lateral=(lateral or baseline),
          verbose=verbose,
          hz=hz,
          speed_limit=SPEED_LIMIT,
          ltl=ltl,)

  if not baseline:
    if agent_name in keras_rl_agents():
      actor = ddpg_get_validation_actor(env, env.action_space.shape[0])
      actor.load_weights('{}_{}_weights_actor.h5f'.format(option,agent_name))
    else:
      actor = default_model(env)
      actor.load_weights('{}_{}_actor.h5f'.format(agent_name,option))

    fast = FastNetwork(actor)
  policy = LateralManualPolicy()

  failures = 0
  ltl = 0
  collisions = 0
  total_duration = 0
  episodes = 0
  total_reward = 0
  for ep in xrange(test_episodes):

    # Generate environments and other settings from numpy with the appropriate
    # random seed.
    if seed is not None:
      np.random.seed(int(seed+ep))
      if verbose: print int(seed+ep), ":",
    f = env.reset()

    R = 0
    for i in xrange(test_length):
      if visualize:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit();

      rw_actor = env._world.actors[0]
      state = rw_actor.state
      if baseline:
        lat_a = policy.evaluate(env._world, state, rw_actor)
        a = [lat_a.a, lat_a.dy]
      elif use_fast:
        a = fast.predict(f)
      else:
        a = actor.predict(np.array([f]))[0]
      (f, r, done, _) = env.step(a)
      R += r

      if done:
        break

      if visualize:
        env.render()

    episodes += 1
    total_duration += (i + 1) * env._world.dt
    total_reward += R
    rw_actor = env._world.actors[0]
    state = rw_actor.state
    prev_state = rw_actor.last_state
    ltl_failed = False
    collided = False
    failed = False
    for condition, wt, name in env._world.conditions:
        if not condition(env._world, state, rw_actor, prev_state):
            if name == "no_collisions":
                collided = True
            elif isinstance(condition, LTLCondition):
                ltl_failed = True
            if wt < 0:
                failed = True
    collisions += collided
    ltl += ltl_failed
    failures += failed

    print "Episode %d: total reward = %f"%(ep, R)

  print "--------------------------------------------------------------"
  print "Failures: ", failures, "/", test_episodes
  print "Collisions: ", collisions, "/", test_episodes
  print "LTL Constraint Violations: ", ltl, "/", test_episodes
  print "Avg Reward: ", (total_reward / episodes), " = ", total_reward, "/", episodes
  print "Avg Duration: ", (total_duration / episodes), " = ", total_duration, "/", episodes
