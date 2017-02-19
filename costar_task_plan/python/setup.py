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
      url='https://github.com/cpaxton/costar_task_plan',
      license='n/a',
      packages=["costar_task_plan",
          "costar_task_plan.abstract",
          "costar_task_plan.core",
          "costar_task_plan.core.utils",
          "costar_task_plan.mcts",
          "costar_task_plan.gym",
          "costar_task_plan.trainers",
          "costar_task_plan.models",
          "costar_task_plan.needle_master",
          "costar_task_plan.costar_world",
          "costar_task_plan.datasets",
          "costar_task_plan.road_world",
          "costar_task_plan.road_world.core",
          "costar_task_plan.road_world.planning",
          "costar_task_plan.road_world.learning",
          "costar_task_plan.road_world.options",
          "costar_task_plan.road_world.ltl",
          "costar_task_plan.tools",
          "costar_task_plan.ros",
          "costar_task_plan.ros.core",
          "costar_task_plan.ros.tools",
          "costar_task_plan.draw",
          "costar_task_plan.traj_opt",
    ],
    scripts=["costar_task_plan/road_world/ltl/ltl_binaries/ltl2ba",
             "costar_task_plan/road_world/ltl/ltl_binaries/ltl2dstar"
    ],
)
