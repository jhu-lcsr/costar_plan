
import os
import sys
import errno
import traceback
import itertools
import six
import glob
import numpy as np
from random import shuffle

import tensorflow as tf
import re
from scipy.ndimage.filters import median_filter
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras._impl.keras.utils.data_utils import _hash_file
import keras
from keras import backend as K

import grasp_utilities


def draw_grasp(axs, grasp_success, center, theta, x_current, y_current, z=2, showTextBox=False, show_center=True):
    widths = [1, 2, 1, 2]
    alphas = [0.25, 0.5, 0.25, 0.5]
    cx, cy = center
    if grasp_success:
        # gray is width, color (purple/green) is plate
        # that's gap, plate, gap plate
        colors = ['gray', 'green', 'gray', 'green']
        success_str = 'pos'
        center_color = 'green'
    else:
        colors = ['gray', 'purple', 'gray', 'purple']
        success_str = 'neg'
        center_color = 'green'

    if show_center:
        axs.scatter(np.array([cx]), np.array([cy]), zorder=z, c=center_color, alpha=0.5, lw=2)
        z += 1

    if showTextBox:
        axs.text(
            cx, cy,
            success_str, size=10, rotation=-np.rad2deg(theta),
            ha="right", va="top",
            bbox=dict(boxstyle="square",
                      ec=(1., 0.5, 0.5),
                      fc=(1., 0.8, 0.8)),
            zorder=z,
            )
        z += 1
    for i, (color, width, alpha, x, y) in enumerate(zip(colors, widths, alphas, x_current, y_current)):
        # axs[h, w].text(example['bbox/x'+str(i)], example['bbox/y'+str(i)], "Point:"+str(i))
        axs.add_line(lines.Line2D(x, y, linewidth=width,
                                  color=color, zorder=z, alpha=alpha))
    return z