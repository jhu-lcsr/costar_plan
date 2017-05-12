

from util import GetAvailableTasks, GetAvailableRobots, GetAvailableAlgorithms

import argparse
import sys

def ParseBulletArgs():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--gui",
                        action="store_true",
                        help="Display Bullet visualization.")
    parser.add_argument("--robot",
                        help="Robot model to load. This will determine the action space.",
                        choices=GetAvailableRobots(),
                        default=GetAvailableRobots()[0])
    parser.add_argument("--task",
                        help="Task to load. This will determine what objects appear in the world as well as the reward function.",
                        choices=GetAvailableTasks(),
                        default=GetAvailableTasks()[0])
    parser.add_argument('--name',
                        help="ROS node name for client (optional; only advanced users need to worry about this)",
                        default="costar_bullet_simulation")
    parser.add_argument('--algorithm',
                        help="Algorithm to use when training.",
                        default=None,
                        choices=GetAvailableAlgorithms())
    parser.add_argument('-l', '--lr', '--learning_rate',
                        help="Learning rate to be used in algorithm.",
                        default=1e-3)
    parser.add_argument('-g','--gamma',
                        help="MDP discount factor gamma. Must be set so that 0 < gamma <= 1. Low gamma decreases significance of future rewards.",
                        default=1.)

    return vars(parser.parse_args())
