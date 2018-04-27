'''
View and Convert dataset lets you look at the video data in a costar stacking dataset.

show a video from an h5f file:

    python view_convert_dataset --path <path/to/data/folder/or/file> --preview True

Convert video from an h5f file into a gif:

python plot_graph --path 'data/' 
    python view_convert_dataset --path <path/to/data/folder/or/file> --preview --convert gif
'''
import argparse
import os
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from costar_models.datasets.image import JpegToNumpy
from costar_models.datasets.image import ConvertJpegListToNumpy
from ctp_integration.stack import verify_stack_success
import pygame



def npy_to_video(npy, filename, fps=10, preview=True, convert='gif'):
    """Convert a numpy array into a gif file at the location specified by filename.
        
        # Arguments

        convert: Default empty string is no conversion, options are gif and mp4.
        preview: pop open a preview window to view the video data.
    """
    # Useful moviepy instructions https://github.com/Zulko/moviepy/issues/159
    # TODO(ahundt) currently importing moviepy prevents python from exiting. Once this is resolved remove the import below.
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(npy), fps)
    if preview:
        # https://stackoverflow.com/a/41771413
        clip.preview()
    if convert == 'gif':
        clip.write_gif(filename)
    elif convert:
        clip.write_videofile(filename)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=os.path.join(os.path.expanduser("~"), '.costar', 'data'),
                        help='path to dataset h5f file or folder containing many files')
    parser.add_argument("--convert", type=str, default='',
                        help='format to convert images to. Default empty string is no conversion, options are gif and mp4.')
    parser.add_argument("--ignore_failure", type=bool, default=False, help='skip grasp failure cases')
    parser.add_argument("--ignore_success", type=bool, default=False, help='skip grasp success cases')
    parser.add_argument("--detect_errors", type=bool, default=False, help='Try to detect errors in the final grasping labels')
    parser.add_argument("--preview", type=bool, default=False, help='pop open a preview window to view the video data')
    parser.add_argument("--fps", type=int, default=10, help='framerate to use for images in frames per second')
    return vars(parser.parse_args())

def main(args,root="root"):

    if '.h5f' in args['path']:
        filenames = [args['path']]
    else:
        filenames = os.listdir(args['path'])

    # Read data
    progress_bar = tqdm(filenames)
    for filename in progress_bar: 
        if filename.startswith('.') or '.h5' not in filename:
            continue
        
        stack_success = 'success' in filename

        if args['ignore_failure'] and 'failure' in filename:
                continue
        if args['ignore_success'] and 'success' in filename:
                continue

        if args['path'] not in filename:
            # prepend the path if it isn't already present
            example_filename = os.path.join(args['path'], filename)
        else:
            example_filename = filename
        progress_bar.set_description("Processing %s" % example_filename)

        data = h5py.File(example_filename,'r')

        # Check if the stack data matches the saved labels according
        # to our verify_stack_success function
        if args['detect_errors']:
            tf2_frames_json = data['all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json']
            print('<<<<<<<<<<<<<<')
            print(tf2_frames_json[-1])
            # get the tf transforms at the last time step
            tf_pose_dict = json.loads(tf2_frames_json[-1])
            verified_stack_success = verify_stack_success(tf_pose_dict)
            print('stack_success ' + str(stack_success) + 'verified_stack_success ' + str(verified_stack_success))
            # if verified_stack_success == stack_success:
            #     continue

        images = list(data['image'])
        # images = ConvertJpegListToNumpy(np.squeeze(images), format='list', data_format='NCHW')
        images = ConvertJpegListToNumpy(np.squeeze(images), format='list')
        # images = ConvertJpegListToNumpy(images)
        # progress_bar.write('numpy gif data shape: ' + str(images.shape))

        # set window labels based on the data label
        if args['preview']:
            if stack_success:
                pygame.display.set_caption('Successful grasp: ' + filename)
            else:
                pygame.display.set_caption('Failure to grasp: ' + filename)

        gif_filename = example_filename.replace('.h5f','.' + args['convert'])
        npy_to_video(images, gif_filename, fps=10, preview=args['preview'], convert=args['convert'])

if __name__ == "__main__":
    args = _parse_args()
    main(args)