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
from costar_task_plan.simulation import ParseBulletArgs
from costar_task_plan.agent import GetAgents, MakeAgent
from costar_models import MakeModel

import time

from ctp_visual.search import *

def sim(args):
    model = MakeModel(taskdef=env.task, **args)
    if 'load_model' in args and args['load_model']:
        model.validate = True
        model.load(env.world)

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
