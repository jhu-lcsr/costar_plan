from abstract import AbstractTaskDefinition
from blocks import BlocksTaskDefinition
from drl_blocks import DRLBlocksTaskDefinition
from clutter import ClutterTaskDefinition
from sorting import SortingTaskDefinition
from oranges import OrangesTaskDefinition
from obstacles import ObstaclesTaskDefinition
from trays import TraysTaskDefinition
from obstructions import ObstructionsTaskDefinition
from sorting2 import Sorting2TaskDefinition
from explore import ExploreTaskDefinition

# TODO(cpaxton): re-enable these
# from mug import MugTaskDefinition
# from rings import RingsTaskDefinition

__all__ = ["AbstractTaskDefinition",
           "BlocksTaskDefinition",
           "DRLBlocksTaskDefinition",
           "ClutterTaskDefinition",
           "SortingTaskDefinition",
           "Sorting2TaskDefinition",
           "OrangesTaskDefinition",
           "ObstaclesTaskDefinition",
           "TraysTaskDefinition",
           "ObstructionsTaskDefinition",
           "ExploreTaskDefinition",
           #"RingsTaskDefinition",
           #"MugTaskDefinition",
           ]
