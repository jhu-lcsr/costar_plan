#!/usr/bin/env bash

# load modules
module load python/2.7.9
module load piplib/1.3.1
module load git

# installing things
git clone git@github.com:cpaxton/costar_plan.git

# python
pip install --user keras numpy
