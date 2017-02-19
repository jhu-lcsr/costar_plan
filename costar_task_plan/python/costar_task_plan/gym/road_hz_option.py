import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from utils import *

import pygame as pg
import task_tree_search.road_world as rw
import task_tree_search.abstract as abstract
import task_tree_search.draw as draw
import task_tree_search.models.adversary as adv

from gym.spaces import Box

'''
Create a simple environment with a car moving down a road. Try to learn not to
crash into the sides of the road, other cars, etc.
'''
class RoadWorldOptionEnv(gym.Env, utils.EzPickle):
  metadata = {'render.modes': ['human']}

  def __init__(self,
      verbose=False,
      speed_limit=5,
      randomize=True,
      planning=False,
      lateral=True,
      ltl=False,
      option="Default",
      adversary="random",
      hz=0):
    self._screen = None

    pg.init()
    self.clock = pg.time.Clock()
    self.lateral = lateral
    self.ltl = ltl
    self.hz = hz
    self.clock = pg.time.Clock()
    self.control_policy = abstract.AbstractPolicy()
    self._option_name = option
    self._randomize = randomize
    self._verbose = verbose
    self._speed_limit = speed_limit
    self._planning = planning

    self.loaded_sprites = False
    self.tried_loading_sprites = False
    self.last_idx = None

    # finally, get the world and set it up
    self._opt = rw.options.makeOption(self._option_name)
    self._adversary = adv.getAdversary(adversary, self._opt)
    self._configure_environment()

    self.observation_space = Box(*self._world.getFeatureBounds())
    self.action_space = Box(np.array([-1,-1]),np.array([1,1]))

  def __del__(self):
    pass

  def _configure_environment(self):
    # update the adversary
    self._adversary.finish() 

    # get a new sample
    params = self._adversary.sample()

    # create associated world
    self._world = self._opt.makeWorld(
      params,
      lateral=self.lateral,
      ltl=self.ltl,
      planning=self._planning,
      speed_limit=self._speed_limit,
      random=self._randomize,
      verbose=self._verbose)
    if not self.tried_loading_sprites:
      try:
        conversion = self._world.meters_to_pixels
        scale = (int(32./40.*3.*conversion), int(50./40.*3.*conversion))
        self.blue = draw.SpriteManager("blue.png",1024,scale,3*np.pi/2)
        self.yellow = draw.SpriteManager("yellow.png",1024,
            (int(1.2*scale[0]), int(1.2*scale[1])),
              3*np.pi/2)
        self.green = draw.SpriteManager("green.png",1024,
            (int(1.2*scale[0]), int(1.2*scale[1])),
              3*np.pi/2)
        self.red = draw.SpriteManager("red.png",1024,scale,3*np.pi/2)
        self.loaded_sprites = True
      except Exception, e:
        print "Warning: could not load sprites.", e
      self.tried_loading_sprites = True
    if self.loaded_sprites:
      self._world.setSprites([self.red, self.blue, self.green, self.yellow])

  def _makeAction(self, action):
    if self.lateral:
      return rw.core.LateralAction(
        a=self._world.a_scale * action[0],
        dy=self._world.dy_scale * action[1])
    else:
      return rw.core.RoadAction(
        a=self._world.a_scale * action[0],
        dw=self._world.dw_scale * action[1])

  # update function: set the actor's policy from the random control
  # or from whatever else we got as an input
  def _step(self, action):
    assert(self._world.features is not None)
    (ok,S0,A0,S1,F1,R) = self._world.tick(self._makeAction(action))
    self._adversary.update(R)
    return (F1, R, not ok,
        {"SO.x":S0.x,
          "S0.y":S0.y,
          "S1.x":S1.x,
          "S1.y":S1.y,
          "S0.v":S0.v,
          "S1.v":S1.v,
          "S0.w":S0.w,
          "S1.w":S1.w,
          "S0.dw":S0.dw,
          "S1.dw":S1.dw,
          "A0.a":A0.a,
          "A0.dy":A0.dy,
          "S1_center_d":S1.lat() - S1.lane_center,
          "S0_center_d":S0.lat() - S0.lane_center,
          "abs(A0.dw)":abs(A0.dw),
          "abs(A0.a)":abs(A0.a),
          "A0.dw":A0.dw,})

  def _reset(self):
    self._configure_environment()
    return self._world.computeFeatures()

  def _render(self, mode='human', close=False):
    # just call world.show()
    
    if mode is 'human':
      if close:
        pg.quit()
      else:
        if self._screen is None:
          self._screen = self._world.getScreen()
        self._world.show(self._screen)
        if self.hz > 0:
          self.clock.tick(self.hz)
