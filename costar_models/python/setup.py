#!/usr/bin/env python

'''
Run this one if you just want a python package.
'''

from distutils.core import setup
setup(name='costar_models',
      version='0.0.1:w',
      description='Deep Policy Learning and Tree Search',
      author='Chris Paxton',
      author_email='cpaxton@jhu.edu',
      url='https://github.com/cpaxton/costar_plan',
      license='n/a',
      packages=["costar_models",
          "costar_models/datasets",
   ],
)
