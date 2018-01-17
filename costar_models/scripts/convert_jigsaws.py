#!/usr/bin/env python

from __future__ import print_function

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


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
    parser.add_argument("--dataset",
                        type=str,
                        default="Suturing",
                        help="dataset to load")
    parser.add_argument("--directory",
                        type=str,
                        default="suturing_data",
                        help="directory to make")
    return parser.parse_args()


def main():
    args = getArgs()
    
    transcriptions_dir = os.path.join(args.dataset, "transcriptions")
    filenames = os.listdir(transcriptions)

    for filename in filenames:

        data = {}
        data["image"] = []
        data["label"] = []
        data["goal_image"] = []
        data["goal_label"] = []
        data["prev_label"] = []

        if filename[0] == ".":
            continue

        txt_file = os.path.join(transcriptions, filename)
        with fin as file(txt_file, "r"):
            res = fin.readLine()
            print(res)

        avi_file = os.path.join(args.dataset,
                "video",
                filename_base,
                "_capture1.avi")
        vid = cv2.VideoCapture(avi_file)
        while vid.isOpened():
            ret, frame = vid.read()

            # TODO: convert from BGR to RGB

            # Return this
            if not ret:
                break

            # Plot 
            plt.imshow(frame)
            plt.show()
            break



if __name__ == "__main__":
    main()
