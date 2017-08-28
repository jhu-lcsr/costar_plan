#!/usr/bin/env bash

# load modules
module load piplib/1.3.1
module load git
module load tensorflow/c62a66b

# installing things
git clone git@github.com:cpaxton/costar_plan.git

# python
pip install --user h5py keras keras-rl sympy matplotlib gmr networkx \
  dtw pypr gym PyPNG pybullet numba


