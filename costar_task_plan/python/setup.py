#!/usr/bin/env python

'''
Run this one if you just want a python package.
'''

from distutils.core import setup
setup(name='costar_task_plan',
      version='0.0.1',
      description='Deep Reinforcement Learning and Tree Search',
      author='Chris Paxton',
      author_email='cpaxton@jhu.edu',
      url='https://github.com/cpaxton/costar_task_search',
      license='n/a',
      packages=["costar_task_search",
          "costar_task_search.abstract",
          "costar_task_search.core",
          "costar_task_search.core.utils",
          "costar_task_search.mcts",
          "costar_task_search.gym",
          "costar_task_search.trainers",
          "costar_task_search.models",
          "costar_task_search.needle_master",
          "costar_task_search.grid_world",
          "costar_task_search.datasets",
          "costar_task_search.road_world",
          "costar_task_search.road_world.core",
          "costar_task_search.road_world.planning",
          "costar_task_search.road_world.learning",
          "costar_task_search.road_world.options",
          "costar_task_search.road_world.ltl",
          "costar_task_search.tools",
          "costar_task_search.ros",
          "costar_task_search.ros.core",
          "costar_task_search.ros.tools",
          "costar_task_search.draw",
          "costar_task_search.traj_opt",
    ],
    scripts=["costar_task_search/road_world/ltl/ltl_binaries/ltl2ba",
             "costar_task_search/road_world/ltl/ltl_binaries/ltl2dstar"
    ],
)
