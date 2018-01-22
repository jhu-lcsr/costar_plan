#!/usr/bin/env python

from __future__ import print_function

from scipy.misc import imresize

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py


def getArgs():
    '''
    Get argument parser and call it to get information from the command line.

    Parameters:
    -----------
    none

    Returns:
    --------
    args: command-line arguments
    '''
    parser = argparse.ArgumentParser(add_help=True,
            description="Process JIGSAWS data.")
    parser.add_argument("base_dir",
                        type=str,
                        help="base directory to load from")
    parser.add_argument("--out_dir", "-o",
                        type=str,
                        default=None,
                        help="directory to make")
    parser.add_argument("--drop_chance", "-d",
                        type=int,
                        default=20)
    return parser.parse_args()


def main():
    args = getArgs()

    # get dataset from path
    basedir_path = os.path.abspath(args.base_dir)
    dataset = os.path.split(basedir_path)[-1]

    if args.out_dir is None:
        args.out_dir = './' + dataset + '_out'

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    trans_dir = os.path.join(args.base_dir, "transcriptions")
    filenames = os.listdir(trans_dir)

    for file_num, filename in enumerate(filenames):

        print(filename)

        if filename[0] == '.':
            continue

        # Get all gestures
        gestures = []

        txt_file = os.path.join(trans_dir, filename)
        with open(txt_file, "r") as f:
            for line in f:
                words = line.split()
                start, end = int(words[0]), int(words[1])
                gesture = int(words[2][1:])
                gestures.append((start, end, gesture))

        #print(gestures)

        filename_base = os.path.splitext(filename)[0]

        avi_file = os.path.join(args.base_dir,
                "video",
                filename_base + "_capture1.avi")
        vid = cv2.VideoCapture(avi_file)

        frames = []
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            # Convert from BGR to RGB
            temp = np.copy(frame[:,:,0])
            frame[:,:,0] = frame[:,:,2]
            frame[:,:,2] = temp
            frames.append(frame)

        goal_images = [] # save goal images for final map

        data = {}
        data["image"] = []
        data["label"] = []
        data["goal_image"] = []
        data["goal_label"] = []
        data["prev_label"] = []
        frame_nums = []

        def get_gesture(frame_num, gestures):
            prev_gesture = 16
            cur_gesture = 16
            next_gesture = None
            goal_frame = False
            final_frame = False
            last_gesture = gestures[-1][2]
            for i, (start, end, gesture) in enumerate(gestures):
                if frame_num < start:
                    return goal_frame, final_frame, prev_gesture, prev_gesture, gesture
                elif frame_num <= end:
                    goal_frame = frame_num == start
                    final_frame = frame_num == end
                    next_gesture = gestures[i+1][2] if i + 1 < len(gestures) else last_gesture
                    return goal_frame, final_frame, prev_gesture, gesture, next_gesture
                else:
                    prev_gesture = gesture

            return False, False, None, None, None

        for i, frame in enumerate(frames):
            frame_num = i + 1

            goal_frame, final_frame, last_gesture, gesture, next_gesture = get_gesture(frame_num, gestures)
            if gesture is None: # check for beyond last
                break
            if gesture == 16: # check for before first
                print("-- skipping frame", i)
                continue

            image = imresize(frame, (96, 128))
            image = image.astype(np.uint8)

            if args.drop_chance > 0 and not goal_frame and not final_frame:
                r = np.random.randint(args.drop_chance)
                if r > 0:
                    continue

            frame_nums.append(frame_num)
            data["image"].append(image)
            data["label"].append(gesture)
            data["goal_label"].append(next_gesture)
            data["prev_label"].append(last_gesture)
            print("i[{}], goal_frame[{}], final_frame[{}], label[{}], goal_lbl[{}], prev_lbl[{}]".format(
                frame_num, goal_frame, final_frame, gesture, next_gesture, last_gesture))

            # save goal_image
            if goal_frame:
                goal_images.append((frame_num, image))

                #if gesture == 8:
                #    print("Gesture =", gesture, "Frame num =", frame_num)
                #    plt.imshow(image)
                #    plt.show()

        # Helper function: look for goals that might work
        # This one will take num and search for entries in the array of goals
        # where those entries occur after this one
        def lookup(num, arr):
            for frame_num, frame in arr:
                if frame_num > num:
                    return frame
            return frame
            
        # Add goal images
        for i in frame_nums:
            frame = lookup(i, goal_images)
            data["goal_image"].append(frame)

        for k, v in data.items():
            data[k] = np.array(v)

        write(args.out_dir, data, file_num, 1)

def write(directory, data, i, r):
    '''
    Write to disk.
    '''
    status = "success" if r > 0. else "failure"
    filename = "example%06d.%s.h5f"%(i, status)
    filename = os.path.join(directory, filename)
    f = h5py.File(filename, 'w')
    for key, value in data.items():
        f.create_dataset(key, data=value)
    f.close()

if __name__ == "__main__":
    main()
