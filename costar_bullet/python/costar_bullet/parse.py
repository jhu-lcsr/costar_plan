

from util import GetAvailableTasks, GetAvailableRobots

import argparse
import sys

def ParseBulletArgs():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--gui",
                        action="store_true",
                        help="Display Bullet visualization.")
    parser.add_argument("--robot",
                        choices=GetAvailableRobots(),
                        default=GetAvailableRobots()[0])
    parser.add_argument("--task",
                        choices=GetAvailableTasks(),
                        default=GetAvailableTasks()[0])

    return args
