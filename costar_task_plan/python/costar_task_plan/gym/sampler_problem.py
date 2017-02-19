import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from utils import *

import pygame as pg
import costar_task_search.road_world as rw
import costar_task_search.mcts as mcts

from gym.spaces import Box

'''
Draw a number of trajectories from the same start position. Learn to generate
actions in an order that minimizes regret.
'''
class SamplerProblemEnv(gym.Env, utils.EzPickle):
  metadata = {'render.modes': ['human']}

  def __init__(self, verbose=False, speed_limit=5, other_cars=True, randomize=True):
    self.screen = None
    pg.init()
    self.clock = pg.time.Clock()
    self.control_policy = rw.RoadControllerPolicy()
    self.F = None
    self.other_cars = other_cars
    self._randomize = randomize
    self._verbose = verbose
    self._speed_limit = speed_limit
    self._configure_environment()

    self.action_space = Box(np.array([-2,-1]),np.array([2,1]))

  def __del__(self):
    pass

  # set up the world with current agents
  def _configure_environment(self):
    RoadWorldType = rw.PlanningRoadWorld
    if self._randomize:
      ry = np.random.randint(100,300)
      sx = np.random.randint(50,500)
      ix = sx + 20
      x = np.random.randint(630)
      y = ry + np.random.randint(20,60)
      v = np.random.rand() * self._speed_limit
    else:
      ry = 230
      x = 0
      v = 0
      sx = 200
      ix = 220
      y = 40
    self.world = RoadWorldType(640,480, loop=False, speed_limit=self._speed_limit, verbose=self._verbose)
    self.T = self.world.getT()
    self.ry = ry
    self.x = x
    self.y = y
    self.v = v
    road = self.world.addRoadHorizontal(ry,80,bidirectional=True)
    stop = self.world.addStopHorizontal(road,sx,20);
    intersection = self.world.addIntersection(ix,ry,80,80)
    self.world.addCondition(rw.inBoxCondition((0,ry,700,80)), -50, "on_route")
    self.world.addCondition(rw.speedLimitCondition(), -50, "under_speed_limit")
    #self.world.addCondition(rw.stopSignCondition(stop.box), -25, "stop_at_stop_signs")
    #self.world.addCondition(rw.collisionCondition(), -100, "no_collisions")
    self.world.addCondition(rw.xltCondition(0), -100, "no_illegal_exit_reverse")
    self.world.addCondition(rw.yltCondition(0), -25, "no_illegal_exit_down")
    self.world.addCondition(rw.ygtCondition(480), -25, "no_illegal_exit_up")
    self.world.addCondition(rw.xgtCondition(640), 150, "finished_route")

    # create the set of features we are going to be using.
    self._make_features(self.world.hroutes[0], stop, intersection)
    self._setup_actors()
    self.world.setFeatures(self.F)
    policies = mcts.DefaultMctsPolicies()
    self.root = mcts.Node(policies, self.world)

    # create observation space
    self.observation_space = Box(*self.world.getFeatureBounds())

  def _make_features(self, route, stop, intersection):
    self.F = rw.SimpleFeatures(route, stop, intersection)

  def _start_viewer(self):
    self.screen = self.world.getScreen()

  # update function: set the actor's policy from the random control
  def _step(self, action):
    assert(self.world.features is not None)
    #(ok,S0,A0,S1,F1,R) = self.world.tick(rw.RoadAction(action[0],action[1]))
    primitive = rw.PlanningRoadAction(action[0],action[1])
    child = self.root.expand(primitive)
    return (child.features(),
        child.reward(),
        len(self.root.children) >= 10,{})

  def _setup_actors(self):
    # delete the dynamic contents of the world and reset all actors
    self.world.reset()

    s0 = rw.RoadState(self.world, x=self.x, y=self.y, v=self.v, w=0)
    S0 = rw.RoadTrajectory([s0])
    car1 = rw.RoadActor(S0, self.control_policy, rw.PlanningRoadDynamics(self.world))
    self.world.addActor(car1)
    if self.other_cars:
      policy = rw.PlanningRoadWorldSpeedPolicy(speed=self.world.speed_limit,
                                               max_acc=1.0,
                                               noise=0.5)
      hz_rw_add_other_cars(self.world,self.T,policy,traj=True)

  def _reset(self):
    if self._randomize:
      self._configure_environment()
    else:
      self._setup_actors()
    return self.world.computeFeatures()

  def _render(self, mode='human', close=False):
    if mode is 'human':
      if close:
        pg.quit()
      else:
        # plot out the different trajectories we've tried so far
        plt.draw()
