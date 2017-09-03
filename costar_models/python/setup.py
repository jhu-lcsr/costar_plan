#!/usr/bin/env python

'''
Run this one if you just want a python package.
'''

from setuptools import setup, find_packages
setup(name='costar_models',
      version='0.0.1',
      description='Deep Policy Learning and Tree Search',
      author='Chris Paxton',
      author_email='cpaxton@jhu.edu',
      url='https://github.com/cpaxton/costar_plan',
      packages=find_packages(),
      #packages=["costar_models",
      #    "costar_models/datasets",
      #],
)
