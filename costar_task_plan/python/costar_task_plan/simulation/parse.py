
from costar_task_plan.agent import GetAgents
from util import GetAvailableTasks, GetAvailableRobots, GetAvailableAlgorithms
from features import GetFeatures, GetAvailableFeatures

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
    parser.add_argument('--start_ros', '--ros', '-R',
                        help="Start as a ROS node.",
                        action="store_true")
    parser.add_argument('--ros_name', '--rn',
                        help="ROS node name for client (optional; only advanced users need to worry about this)",
                        default="costar_bullet_simulation")
    parser.add_argument('--agent',
                        help="Algorithm to use when training.",
                        default="null",
                        choices=GetAgents())
    parser.add_argument('-l', '--lr', '--learning_rate',
                        help="Learning rate to be used in algorithm.",
                        default=1e-3)
    parser.add_argument('-g', '--gamma',
                        help="MDP discount factor gamma. Must be set so that 0 < gamma <= 1. Low gamma decreases significance of future rewards.",
                        default=1.)
    parser.add_argument('-o','--option',
                        help="Specific sub-option to train. Exact list depends on the chosen task.",
                        default=None)
    parser.add_argument('-p','--plot_task','--pt',
                        help="Display a plot of the chosen task and exit.",
                        action="store_true")
    parser.add_argument('-s', '--save',
                        help="Save training data",
                        action="store_true")
    parser.add_argument('-c', '--capture',
                        help="Capture images as a part of the training data",
                        action="store_true")
    parser.add_argument('-d', '--directory',
                        help="Directory to store data from trials",
                        default="./")
    parser.add_argument('--show_images',
                        help="Display images from cameras for debugging.",
                        action="store_true")
    parser.add_argument('--randomize_color',
                        help="Randomize colors for the loaded robot.",
                        action="store_true")
    parser.add_argument('--features',
                        help="Specify feature function",
                        default="null")

    return vars(parser.parse_args())
