
from costar_task_plan.agent import GetAgents
from costar_task_plan.models import GetModels

from util import GetAvailableTasks, GetAvailableRobots, GetAvailableAlgorithms
from features import GetFeatures, GetAvailableFeatures

import argparse
import sys

_desc = """
Start the CTP bullet simulation tool. This will allow you to generate data, run
reinforcement learning algorithms, and test models and algorithms.
"""
_epilog = """
Some example tasks are "blocks," which generates a set of colored blocks. The
robot should pick a specific block up and put it in the center of the
workspace. In "tower," we again generate a set of colored blocks. This time the
robot should pick them all up and stack them.
"""


def ParseBulletArgs():
    parser = argparse.ArgumentParser(add_help=True,
                                     description=_desc, epilog=_epilog)
    parser.add_argument("--gui",
                        action="store_true",
                        help="Display Bullet visualization.")
    parser.add_argument("--opengl2",
                        action="store_true",
                        help="Activate docker opengl2 mode")
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
    parser.add_argument('-L', '--lr', '--learning_rate',
                        help="Learning rate to be used in algorithm.",
                        type=float,
                        default=1e-3)
    parser.add_argument('-g', '--gamma',
                        help="MDP discount factor gamma. Must be set so that 0 < gamma <= 1. Low gamma decreases significance of future rewards.",
                        default=1.)
    parser.add_argument('-o', '--option',
                        help="Specific sub-option to train. Exact list " + \
                             "depends on the chosen task. [NOT CURRENTLY " + \
                             "IMPLEMENTED]",
                        default=None)
    parser.add_argument('-p', '--plot_task', '--pt',
                        help="Display a plot of the chosen task and exit.",
                        action="store_true")
    parser.add_argument('-s', '--save',
                        help="Save training data",
                        action="store_true")
    parser.add_argument('-l', '--load',
                        help="Load training data from file." + \
                        " Use in conjunction with save to append to" + \
                        " a training data file.",
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
                        default="null",
                        choices=GetAvailableFeatures())
    parser.add_argument('--profile',
                        help='Run cProfile on agent',
                        action="store_true")
    parser.add_argument('-i', '--iter',
                        help='Number of iterations to run',
                        default=100,
                        type=int)
    parser.add_argument('-m', '--model',
                        help="Name of NN model to learn.",
                        default=None,
                        choices=GetModels())
    parser.add_argument('-b','--batch_size',
                        help='Batch size to use in the model',
                        default=32,
                        type=int)
    parser.add_argument('-e','--epochs',
                        help="Number of epochs",
                        type=int,
                        default=1000,)
    parser.add_argument('--data_file_name',
                        help="File name for data archive.",
                        default='data.npz')
    parser.add_argument('--model_descriptor',
                        help="model description for use with save/load file",
                        default="model")
    parser.add_argument("--optimizer","--opt",
                        help="optimizer to use with learning",
                        default="adam")
    parser.add_argument("-z", "--zdim", "--noise_dim",
                        help="size of action parameterization",
                        default=16)

    return vars(parser.parse_args())
