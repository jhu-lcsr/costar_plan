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
      url='https://github.com/cpaxton/task_tree_search',
      license='n/a',
      packages=["task_tree_search",
          "task_tree_search.abstract",
          "task_tree_search.core",
          "task_tree_search.core.utils",
          "task_tree_search.mcts",
          "task_tree_search.gym",
          "task_tree_search.trainers",
          "task_tree_search.models",
          "task_tree_search.needle_master",
          "task_tree_search.grid_world",
          "task_tree_search.datasets",
          "task_tree_search.road_world",
          "task_tree_search.road_world.core",
          "task_tree_search.road_world.planning",
          "task_tree_search.road_world.learning",
          "task_tree_search.road_world.options",
          "task_tree_search.road_world.ltl",
          "task_tree_search.tools",
          "task_tree_search.ros",
          "task_tree_search.ros.core",
          "task_tree_search.ros.tools",
          "task_tree_search.draw",
          "task_tree_search.traj_opt",
    ],
    scripts=["task_tree_search/road_world/ltl/ltl_binaries/ltl2ba",
             "task_tree_search/road_world/ltl/ltl_binaries/ltl2dstar"
    ],
)
