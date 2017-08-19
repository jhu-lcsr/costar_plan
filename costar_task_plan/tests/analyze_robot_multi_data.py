#!/usr/bin/env python

import costar_task_plan as ctp
import numpy as np

from costar_task_plan.models.split import *

# Get all the data
sim = ctp.simulation.CostarBulletSimulation(robot="ur5", task="blocks", gui=True)
data = np.load('large.npz')

print sim.task.task.indices
for i in data['label']:
    if i not in sim.task.task.indices:
        print i, "not in idx"
    else:
        print sim.task.task.index(i), i

key = "align('goal=green_block')"

img1 = data['features'][data['label']==key]
print len(img1)
print img1.shape

import matplotlib.pyplot as plt
j = 0
for i in xrange(0,250,10):
    plt.subplot(5,5,j+1)
    plt.imshow(img1[j])
    j += 1
    print i, j
plt.show()

