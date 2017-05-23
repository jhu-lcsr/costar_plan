
import numpy as np
import pygame.image as image
import pygame.transform as tf

'''
Sprite manager
'''
class SpriteManager:
  def __init__(self, filename, num=256, scale=None, base=0):
    self.filename = filename
    self.image = image.load(filename)
    self.rotated = []
    self.base = base

    if scale is not None:
      self.image = tf.scale(self.image, scale)

    for i in xrange(num):
      angle = float(i)/num * 360.0
      img = tf.rotate(self.image, angle)
      self.rotated.append((img, img.get_rect()))

  def get(self,radians):
    idx = int((-radians + self.base) / (2*np.pi) * len(self.rotated))
    return self.rotated[idx % len(self.rotated)]
