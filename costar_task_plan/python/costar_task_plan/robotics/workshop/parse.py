from argparse import *
import sys

_desc = """Start the CTP data collection tool for workshop assistant tasks. This is meant to be used in conjunction with costar_stack."""
_epilog = """Data will be dumped to a set of .npz archives."""

def GetWorkshopParser():
    parser = argparse.ArgumentParser(add_help=True, parents=[], description=_desc, epilog=_epilog)
    parser.add_argument("--hz", type=int, default=10, help="Rate at which to save images")
    parser.add_argument("--width", type=int, default=128, help="Width of image output to save")
    parser.add_argument("--height", type=int, default=128, help="Height of image output to save")
    return parser

def ParseWorkshopArgs():
    parser = GetWorkshopParser()
    return vars(parser.parse_args())
