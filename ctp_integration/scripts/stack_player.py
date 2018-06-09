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
import glob

import h5py
import bokeh
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider
from bokeh.models import Button
from bokeh.models.widgets import TextInput

import numpy as np
import io
from PIL import Image
import argparse
from functools import partial

parser = argparse.ArgumentParser(description='Process additional parameters for stack player')

parser.add_argument('--preprocess-inplace', type = str, action = 'store', default = 'None', 
                    help='Options include gripper_action, used to generate gripper_action_label and index of frame before next action')
parser.add_argument('--data', type = str, action = 'store', default = '/costar_task_planning_stacking_dataset_v0.1', 
                    help='directory path containing the data')

args = parser.parse_args()

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

def generate_holo_map(rgb_images,height, width):
    frame_map = {}
    for i, image in enumerate(rgb_images):
        hv_rgb = hv.RGB(image)
        shape = image.shape
        frame_map[i] = hv_rgb
    holomap = hv.HoloMap(frame_map)
    holomap = holomap.options(width=width, height=height)
    return holomap

def process_image(file_path):
    data = h5py.File(file_path, 'r')
    rgb_images = list(data['image'])
    frame_indices = np.arange(len(rgb_images))
    gripper_status = list(data['gripper'])
    action_status = list(data['label'])
    gripper_action_goal_idx = []

    #generate new action labels and goal action indices
    if(args.preprocess_inplace == 'gripper_action'):
        unique_actions, indices= np.unique(action_status, return_index = True)
        unique_actions = [action_status[index] for index in sorted(indices)]
        action_ind = 0
        gripper_action_label = action_status[:]
        for i in range(len(gripper_status)):
            if (gripper_status[i]>0.1 and gripper_status[i-1]<0.1) or(gripper_status[i]<0.5 and gripper_status[i-1]>0.5):
                action_ind+=1
                print(i)
                gripper_action_goal_idx.append(i)
            gripper_action_label[i]=unique_actions[action_ind]

    rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
    return rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx

def load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width):
    # load the gripper data
    gripper_data = hv.Table({'Gripper': gripper_status, 'Frame': frame_indices},
                            ['Gripper', 'Frame'])
    gripper_curves = gripper_data.to.curve('Frame', 'Gripper')
    gripper_curves = gripper_curves.options(width=width, height=height//4)
    gripper_plot = renderer.get_plot(gripper_curves)

    # load the action data
    action_data = hv.Table({'Action': action_status, 'Frame': frame_indices},
                           ['Action', 'Frame'])
    action_curves = action_data.to.curve('Frame', 'Action')
    action_curves = action_curves.options(width=width, height=height//4)
    action_plot = renderer.get_plot(action_curves)

    # load the gripper action label

    gripper_action_data = hv.Table({'Gripper Action': gripper_action_label, 'Frame': frame_indices},
                           ['Gripper Action', 'Frame'])
    gripper_action_curves = gripper_action_data.to.curve('Frame', 'Gripper Action')
    gripper_action_curves = gripper_action_curves.options(width=width, height=height//4)
    gripper_action_plot = renderer.get_plot(gripper_action_curves)

    return gripper_plot, action_plot, gripper_action_plot



renderer = hv.renderer('bokeh')

#example_filename = "C:/Users/Varun/JHU/LAB/Projects/costar_task_planning_stacking_dataset_v0.1/2018-05-23-20-18-25_example000002.success.h5f"
file_name_list = glob.glob(args.data)



rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx = process_image(file_name_list[0])
print('images loaded')
# Declare the HoloViews object
start = 0
end = len(rgb_images)
# TODO(ahundt) resize image, all of this size code had no effect
width = int(640*1.5)
height = int(480*1.5)
if end > 0:
    height = rgb_images[0].shape[0]
    width = rgb_images[0].shape[1]

# hv.opts(plot_width=width, plot_height=height)

holomap = generate_holo_map(rgb_images,height,width)

# Convert the HoloViews object into a plot
plot = renderer.get_plot(holomap)
# bokeh.plotting.curplot().plot_width=800
# plot.plot_width=width

print('holomap loaded')

gripper_plot, action_plot, gripper_action_plot = load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width)

def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year

def slider_update(attrname, old, new):
    plot.update(slider.value)

slider = Slider(start=start, end=end, value=0, step=1, title="Frame", width=width)
slider.on_change('value', slider_update)

def animate():
    if button.label == ' Play':
        button.label = ' Pause'
        curdoc().add_periodic_callback(animate_update, 10)
    else:
        button.label = ' Play'
        curdoc().remove_periodic_callback(animate_update)
def next_image(files,action):
    global file_textbox, button, button_next, button_prev, count
    print("next clicked")
    file_textbox.value = "Processing..."
    renderer = hv.renderer('bokeh')
    if action == 'next':
        count=(count + 1) % len(files)
    else:
        count=(count - 1) % len(files)
    #print("it ", iterator)
    file_name = files[count]
    rgb_images, frame_indices, gripper_status, action_status, gripper_action_label, gripper_action_goal_idx = process_image(file_name)
    print("image loaded")
    print("action goal idx", gripper_action_goal_idx)
    height = rgb_images[0].shape[0]
    width = rgb_images[0].shape[1]
    start = 0
    end = len(rgb_images)
    print(end)

    def slider_update(attrname, old, new):
        plot.update(slider.value)

    slider = Slider(start=start, end=end, value=0, step=1, title="Frame", width=width)
    slider.on_change('value', slider_update)

    holomap = generate_holo_map(rgb_images,height,width)
    print("generated holomap")
    plot = renderer.get_plot(holomap)
    print("plot rendered")
    gripper_plot, action_plot, gripper_action_plot = load_data_plot(renderer, frame_indices, gripper_status, action_status, gripper_action_label, height, width)
    print("plot loaded..")
    plot_list = [
    [plot.state],
    [gripper_plot.state],
    [action_plot.state]]

    widget_list = [[slider, button, button_prev, button_next],[file_textbox]]

    if args.preprocess_inplace == "gripper_action":
        plot_list.append([gripper_action_plot.state])
    layout_child = layout(plot_list+widget_list, sizing_mode='fixed')
    curdoc().clear()
    file_textbox.value = file_name.split("\\")[-1]
    #curdoc().remove_root(layout_child)
    #layout_root.children[0] = layout_child
    curdoc().add_root(layout_child)

#iterator = iter(file_name_list)
count = 1

button = Button(label=' Play', width=60)
button.on_click(animate)

button_next = Button(label = 'Next', width =60)
button_next.on_click(partial(next_image, files=file_name_list, action = 'next'))
button_prev = Button(label = 'Prev', width =60)
button_prev.on_click(partial(next_image, files=file_name_list, action = 'prev'))


# https://bokeh.pydata.org/en/latest/docs/reference/models/widgets.inputs.html
# TODO(ahundt) switch to AutocompleteInput with list of files
file_textbox = TextInput(value=file_name_list[0].split('\\')[-1], width=width)

# TODO(ahundt) load another file when it changes
# def textbox_update(attrname, old, new):
#     plot.update(slider.value)

# file_textbox.on_change(textbox_update)

# Combine the bokeh plot on plot.state with the widgets
plot_list = [
    [plot.state],
    [gripper_plot.state],
    [action_plot.state]]

widget_list = [[slider, button, button_prev, button_next],[file_textbox]]

if args.preprocess_inplace == "gripper_action":
    plot_list.append([gripper_action_plot.state])

layout_root = layout(plot_list+widget_list, sizing_mode='fixed')

#print(layout)



curdoc().add_root(layout_root)