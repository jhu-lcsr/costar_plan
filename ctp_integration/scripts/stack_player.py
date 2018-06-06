# -*- coding: utf-8 -*-
"""
An example of a simple player widget animating an Image demonstrating
how to connnect a simple HoloViews plot with custom widgets and
combine them into a bokeh layout.

The app can be served using:

    bokeh serve --show player.py

"""
import numpy as np
import holoviews as hv
import os

import h5py
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
from bokeh.models.widgets import TextInput
import numpy as np
import io
from PIL import Image

def GetJpeg(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="JPEG", quality=80)
    return output.getvalue()

def GetPng(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(img)
    output = io.BytesIO()
    # enabling optimized file size
    # increases saving time to ~0.4 seconds per image.
    #im.save(output, format="PNG", optimize=True)
    im.save(output, format="PNG")
    return output.getvalue()

def JpegToNumpy(jpeg):
    stream = io.BytesIO(jpeg)
    im = Image.open(stream)
    return np.asarray(im, dtype=np.uint8)

def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC'):
    """ Convert a list of binary jpeg or png files to numpy format.

    # Arguments

    data: a list of binary jpeg images to convert
    format: default 'numpy' returns a 4d numpy array,
        'list' returns a list of 3d numpy arrays
    """
    length = len(data)
    imgs = []
    for raw in data:
        img = JpegToNumpy(raw)
        if data_format == 'NCHW':
            img = np.transpose(img, [2, 0, 1])
        imgs.append(img)
    if format == 'numpy':
        imgs = np.array(imgs)
    return imgs


renderer = hv.renderer('bokeh')

example_filename = os.path.expanduser('~/datasets/costar_task_planning_stacking_dataset_v0.1/2018-05-23-19-08-41_example000008.success.h5f')

data = h5py.File(example_filename, 'r')
print('dataset open')
rgb_images = list(data['image'])

rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
print('images loaded')
# Declare the HoloViews object
start = 0
end = len(rgb_images)
hmap = hv.HoloMap({i: hv.RGB(image) for i, image in enumerate(rgb_images)})

print('hmap loaded')
# Convert the HoloViews object into a plot
plot = renderer.get_plot(hmap)

def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year

def slider_update(attrname, old, new):
    plot.update(slider.value)

slider = Slider(start=start, end=end, value=0, step=1, title="Year")
slider.on_change('value', slider_update)

def animate():
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(animate_update)

button = Button(label='► Play', width=60)
button.on_click(animate)
# dir = TextInput(label=example_filename)

# Combine the bokeh plot on plot.state with the widgets
layout = layout([
    [plot.state],
    [slider, button],
], sizing_mode='fixed')

curdoc().add_root(layout)
