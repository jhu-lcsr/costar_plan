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
import matplotlib.pyplot as plt
# https://github.com/tanyaschlusser/array2gif
# from array2gif import write_gif

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

# from costar_models.datasets.image import JpegToNumpy
# from costar_models.datasets.image import ConvertImageListToNumpy
import pygame
import io
from PIL import Image
import moviepy
import moviepy.editor as mpye
# import skimage

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
    parser.add_argument("--ignore_error", type=bool, default=False, help='skip grasp attempts that are both failures and contain errors')
    parser.add_argument("--preview", action='store_true', help='pop open a preview window to view the video data')
    parser.add_argument("--gripper", type=bool, default=False, help='print gripper data channel')
    parser.add_argument("--depth", type=bool, default=True, help='process depth data')
    parser.add_argument("--rgb", type=bool, default=True, help='process rgb data')
    parser.add_argument("--fps", type=int, default=10, help='framerate to process images in frames per second')
    parser.add_argument("--matplotlib", type=bool, default=False,
                        help='preview data with matplotlib, slower but you can do pixel lookups')
    parser.add_argument("--print", type=str, default='',
                        help=('Comma separated list of data channels to convert to a list and print as a string.'
                              'Options include: label, gripper, pose, nsecs, secs, q, dq, labels_to_name, all_tf2_frames_as_yaml, '
                              'all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json, and more. See collector.py for more key strings.'))
    parser.add_argument('--preprocess_inplace', type=str, action='store', default='',
                        help="""Currently the only option is gripper_action, which generates new labels
                                gripper_action_label and gripper_action_goal_idx based on the timestep at which the gripper opened and closed,
                                and inserts them directly into the hdf5 file.
                             """)

    return vars(parser.parse_args())

def draw_matplotlib(depth_images, fps):
    for image in tqdm(depth_images):
        plt.imshow(image)
        plt.pause(1.0 / fps)
        plt.draw()

def main(args, root="root"):

    if '.h5f' in args['path']:
        filenames = [args['path']]
    else:
        filenames = os.listdir(os.path.expanduser(args['path']))

    # Read data
    progress_bar = tqdm(filenames)
    for filename in progress_bar:
        # skip certain files based on command line parameters
        if filename.startswith('.') or '.h5' not in filename:
            continue
        if args['ignore_error'] and 'error' in filename:
            progress_bar.write('Skipping example containing errors: ' + filename)
            continue
        if args['ignore_failure'] and 'failure' in filename:
            progress_bar.write('Skipping example containing failure: ' + filename)
            continue
        if args['ignore_success'] and 'success' in filename:
            progress_bar.write('Skipping example containing success: ' + filename)
            continue

        if args['path'] not in filename:
            # prepend the path if it isn't already present
            example_filename = os.path.join(args['path'], filename)
        else:
            example_filename = filename

        description = "Processing %s" % example_filename
        progress_bar.set_description(description)
        pygame.display.set_caption(description)

        mode = 'r'
        if len(args['preprocess_inplace']) > 0:
            mode = 'r+'

        # open the file
        try:
            with h5py.File(example_filename, mode) as data:
                fps = args['fps']
                # check if the data is there to load
                load_depth = args['depth'] and 'depth_image' in data and len(data['depth_image']) > 0
                load_rgb = args['rgb'] and 'image' in data and len(data['depth_image']) > 0

                if args['gripper']:
                    # print the gripper data channel
                    progress_bar.write(filename + ': ' + str(list(data['gripper'])))

                if args['print']:
                    # comma separated keys from which to create a list and
                    # print string values from the h5f file
                    data_to_print = args['print'].split(',')
                    for data_str in data_to_print:
                        progress_bar.write(filename + ' ' + data_str + ': ' + str(list(data[data_str])))

                if args['preprocess_inplace'] == 'gripper_action':
                    # generate new action labels based on when the gripper opens and closes
                    if "gripper_action_label" not in list(data.keys()):
                        gripper_action_label, gripper_action_goal_idx = generate_gripper_action_label(data)
                        # add the new action label and goal indices based on when the gripper opens/closes
                        data['gripper_action_label'], data['gripper_action_goal_idx'] = np.array(gripper_action_label), np.array(gripper_action_goal_idx)
                    else:
                        progress_bar.write(
                            'skipping file, gripper_action_label is already present: ' +
                            str(example_filename) + ' gripper_action_goal_idx: ' + str(list(data['gripper_action_goal_idx'])))

                    # skip other steps like video viewing,
                    # so this conversion runs 1000x faster
                    continue

                # Video display and conversion
                try:
                    if load_depth:
                        depth_images = list(data['depth_image'])
                        depth_images = ConvertImageListToNumpy(np.squeeze(depth_images), format='list')
                        if args['matplotlib']:
                            draw_matplotlib(depth_images, fps)
                        depth_clip = mpye.ImageSequenceClip(depth_images, fps=fps)
                        clip = depth_clip
                    if load_rgb:
                        rgb_images = list(data['image'])
                        rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='list')
                        if args['matplotlib']:
                            draw_matplotlib(rgb_images, fps)
                        rgb_clip = mpye.ImageSequenceClip(rgb_images, fps=fps)
                        clip = rgb_clip

                    if load_depth and load_rgb:
                        clip = mpye.clips_array([[rgb_clip, depth_clip]])

                    if args['preview']:
                        clip.preview()

                    save_filename = example_filename.replace('.h5f', '.' + args['convert'])
                    if 'gif' in args['convert']:
                        clip.write_gif(save_filename, fps=fps)
                    elif args['convert']:
                        clip.write_videofile(save_filename, fps=fps)
                except KeyboardInterrupt as ex:
                    progress_bar.write('Keyboard interrupt detected. Exiting')
                    break
                except Exception as ex:
                    progress_bar.write(
                        'Warning: Skipping File. Exception encountered while processing ' + example_filename +
                        ' please edit the code to debug the specifics: ' + str(ex))

        except IOError as ex:
            progress_bar.write(
                'Error: Skipping file due to IO error when opening ' +
                example_filename + ': ' + str(ex))
            continue


def generate_gripper_action_label(data):
    """ generate new action labels and goal action indices based on the gripper open/closed state

    This performs an in place modification of the data argument sets 'gripper_action_label' and 'gripper_action_goal_idx' in the data argument.

    Use stack_player.py to visualize the dataset, that makes it easier to understand what this data is.

    # Returns

        gripper_action_label, gripper_action_goal_idx

        numpy array containing integer label values,
        and numpy array containing integer goal timestep indices.
    """
    gripper_status = list(data['gripper'])
    action_status = list(data['label'])
    gripper_action_goal_idx = []
    unique_actions, indices = np.unique(action_status, return_index=True)
    unique_actions = [action_status[index] for index in sorted(indices)]
    action_ind = 0
    gripper_action_label = action_status[:]
    for i in range(len(gripper_status)):
        if (gripper_status[i] > 0.1 and gripper_status[i-1] < 0.1) or (gripper_status[i] < 0.5 and gripper_status[i-1] > 0.5):
            action_ind += 1
            # print(i)
            gripper_action_goal_idx.append(i)

        # For handling error files having improper data
        if len(unique_actions) <= action_ind or len(gripper_action_label) <= i:
            break
        else:
            gripper_action_label[i] = unique_actions[action_ind]

    return gripper_action_label, gripper_action_goal_idx


if __name__ == "__main__":
    args = _parse_args()
    main(args)