import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from utils import *

import pygame as pg
import costar_task_plan.road_world as rw
import costar_task_plan.abstract as abstract

from gym.spaces import Box

'''
Create a simple environment with a car moving down a road. Try to learn not to
crash into the sides of the road, other cars, etc.
'''
class SimpleRoadWorldEnv(gym.Env, utils.EzPickle):
  metadata = {'render.modes': ['human']}

  # NOTE: yes, this is bad coding. But in the interests of fast development,
  # right now there are IF-statements in configure_environment() that switch
  # between control, planning, and planning with lateral movement primitives.
  def __init__(self, verbose=False, speed_limit=5,
      other_cars=True,
      randomize=True,
      planning=False,
      lateral=False,
      enforce_stop=False):
    self.screen = None
    pg.init()
    self.clock = pg.time.Clock()
    self.control_policy = abstract.AbstractPolicy()
    self.F = None
    self.other_cars = other_cars
    self._randomize = randomize
    self._verbose = verbose
    self._speed_limit = speed_limit
    self._planning = planning
    self._lateral = lateral
    self._enforce_stop = enforce_stop
    self._configure_environment()

    self.action_space = Box(np.array([-2,-1]),np.array([2,1]))

  def __del__(self):
    pass

  # set up the world with current agents
  def _configure_environment(self):
    if self._lateral and self._planning:
      #print "[CONFIGURATION] Using LATERAL PLANNING world!"
      RoadWorldType = rw.planning.LateralPlanningRoadWorld
    elif self._planning:
      #print "[CONFIGURATION] Using PLANNING world!"
      RoadWorldType = rw.planning.PlanningRoadWorld
    elif self._lateral:
      #print "[CONFIGURATION] Using LATERAL world!"
      RoadWorldType = rw.core.LateralRoadWorld
    else:
      RoadWorldType = rw.core.RoadWorld
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
    self.ix = ix
    road = self.world.addRoadHorizontal(ry,80,bidirectional=False)
    stop = self.world.addStopHorizontal(road,sx,20);
    intersection = self.world.addIntersection(ix,ry,80,80)
    road2 = self.world.addRoadVertical(ix,80,bidirectional=False)
    stop2 = self.world.addStopVertical(road,ry-20,20);

    if not self._planning:
      self.world.addCondition(rw.CollisionCondition(), -100, "no_collisions")
    else:
      self.world.addTrajectoryCondition(rw.planning.trajectoryCollisionCondition(), -100, "no_collisions")

    self.world.addCondition(rw.InBoxCondition((0,ry,700,80)), -50, "on_route")
    self.world.addCondition(rw.SpeedLimitCondition(), -50, "under_speed_limit")
    self.world.addCondition(rw.xltCondition(0), -100, "no_illegal_exit_reverse")
    self.world.addCondition(rw.yltCondition(0), -25, "no_illegal_exit_down")
    self.world.addCondition(rw.ygtCondition(480), -25, "no_illegal_exit_up")
    self.world.addCondition(rw.xgtCondition(640), 150, "finished_route")

    is_stopped = rw.IsStoppedCondition()
    is_in_stop_region = rw.InBoxCondition(stop.box)
    is_stopped_in_stop_region = rw.AndCondition(is_stopped, is_in_stop_region)
    has_stopped_in_stop_region = rw.HasCondition(is_stopped_in_stop_region)
    has_in_stop_region = rw.HasCondition(is_in_stop_region)
    not_in_stop_region = rw.NotCondition(is_in_stop_region)
    self.world.addPredicate("in_intersection", rw.InBoxCondition(intersection.box))
    self.world.addPredicate("on_learner_route", rw.InBoxCondition((0,ry,660,80)))
    self.world.addPredicate("not_in_stop_region", not_in_stop_region)
    self.world.addPredicate("has_in_stop_region", has_in_stop_region)
    self.world.addPredicate("has_stopped_in_stop_region", has_stopped_in_stop_region)

    if self._enforce_stop:
      self.world.addCondition(rw.StoppedInStopRegionCondition(), -50, "stop_in_stop_region")

    # create the set of features we are going to be using.
    self._make_features(self.world.hroutes[0], stop, intersection)
    self._setup_actors()
    self.world.setFeatures(self.F)

    # create observation space
    self.observation_space = Box(*self.world.getFeatureBounds())

  def _make_features(self, route, stop, intersection):
    self.F = rw.SimpleFeatures(route, stop, intersection)

  def _start_viewer(self):
    self.screen = self.world.getScreen()

  def _makeAction(self, action):
    if self._lateral:
      return rw.LateralAction(a=action[0], dy=action[1])
    else:
      return rw.RoadAction(action[0],action[1])

  # update function: set the actor's policy from the random control
  def _step(self, action):
    assert(self.world.features is not None)
    (ok,S0,A0,S1,F1,R) = self.world.tick(self._makeAction(action))
    return (F1, R, not ok,
        {"SO.x":S0.x,
          "S0.y":S0.y,
          "S1.x":S1.x,
          "S1.y":S1.y,
          "A0.a":A0.a,
          "A0.dw":A0.dw,})

  def _setup_actors(self):
    # delete the dynamic contents of the world and reset all actors
    self.world.reset()

    if not self._lateral:
      s0 = rw.RoadState(self.world, x=self.x, y=self.y, v=self.v, w=0)
      if self._planning:
        S0 = rw.planning.RoadTrajectory([s0])
        car1 = rw.core.RoadActor(S0, self.control_policy,
            rw.planning.PlanningRoadDynamics(self.world))
      else:
        car1 = rw.core.RoadActor(s0, self.control_policy, self.T)
    else:
      s0 = rw.core.LateralState(self.world, x=self.x, y=self.y, v=self.v, w=0)
      if self._planning:
        S0 = rw.planning.LateralTrajectory([s0])
        car1 = rw.core.RoadActor(S0, self.control_policy,
            rw.planning.PlanningLateralDynamics(self.world))
      else:
        car1 = rw.core.RoadActor(s0,dynamics=self.T)
    self.world.addActor(car1)
    if self.other_cars:
      hz_rw_add_other_cars(self.world,
          self.T,
          rx=self.ix,
          lateral=self._lateral,
          traj=self._planning)

  def _reset(self):
    if self._randomize:
      self._configure_environment()
    else:
      self._setup_actors()
    return self.world.computeFeatures()

  def _render(self, mode='human', close=False):
    # just call world.show()
    
    if mode is 'human':
      if close:
        pg.quit()
      else:
        if self.screen is None:
          self._start_viewer()
        self.world.show(self.screen)
