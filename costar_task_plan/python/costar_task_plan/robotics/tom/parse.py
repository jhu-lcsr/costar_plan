
import argparse

def ParseTomArgs():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--regenerate_models",
                        action="store_true",
                        help="Load data from bags and generate new DMPs, feature models.")
    parser.add_argument('-l','--loop',
                        action="store_true",
                        help="loop")
    parser.add_argument('-e','--execute',
                        action="store_true",
                        help="execute plan by sending points and gripper commands")

    return parser.parse_args()
