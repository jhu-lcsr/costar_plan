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

# -----------------------------------------------------------
# Unused, but retained as an example:
#export CTP_MODELS=$HOME/costar_plan/costar_models/python
#cd $CTP_MODELS
#python setup.py install --prefix=$HOME/my-python-modules 
# Default path:
# $HOME/.local/lib/python3.6/site-packages/$PATH

# -----------------------------------------------------------
# For now we have a constant location where we create these files
export CTP_MODELS=$HOME/costar_plan/costar_models/python
cd $CTP_MODELS
python setup.py install --prefix=$HOME/my-python-modules 
cd $HOME

mkdir -p $HOME/privatemodules/
ln -s $HOME/costar_plan/setup/costar_models_module $HOME/privatemodules/costar_models

