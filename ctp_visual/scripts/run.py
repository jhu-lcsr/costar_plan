#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Import models and dataset stuff
from costar_models import *
from costar_models.sampler2 import PredictionSampler2
from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.npy_generator import NpzGeneratorDataset
from costar_models.datasets.h5f_generator import H5fGeneratorDataset

# ----------------------------------------------------------------
# Import bullet stuff
from costar_task_plan.simulation import ParseBulletArgs, BulletSimulationEnv
from costar_task_plan.agent import GetAgents, MakeAgent
from costar_models import MakeModel

import time

from ctp_visual.search import VisualSearch

def sim(args):
    env = BulletSimulationEnv(**args)
    model = MakeModel(taskdef=env.task, **args)
    model.validate = True
    model.load(env.world)
    search = VisualSearch(model)
    features = env.reset()
    I = np.expand_dims(features[0], axis=0) / 255.
    h = model.encode(I)
    Id = model.decode(h)
    plt.figure()
    search(I, iter=1, depth=5, draw=True)
    plt.show()
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(I[0])
    plt.subplot(1,3,2)
    plt.imshow(np.mean(h[0],axis=-1))
    plt.subplot(1,3,3)
    plt.imshow(Id[0])
    plt.show()

def dataset(args):
    pass

def main(args):
    pass

def _profile(args):
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)

if __name__ == "__main__":
    args = ParseBulletArgs()
    print(args)
    _profile(args)
